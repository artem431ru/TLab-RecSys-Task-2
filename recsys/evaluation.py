"""Offline metrics, user-cluster bootstrap, and hypothetical rollout."""

from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SeqDataset, collate
from .models import last_logits, mask_seen

@torch.no_grad()
def evaluate_rm(rm, rows, args, device, n, pop):
    dl = DataLoader(SeqDataset(rows), batch_size=args.batch, shuffle=False,
                    collate_fn=lambda b: collate(b, n, n + 1, args.ctx))
    yh, yt = [], []
    for ids, mask, _, y, r, _, _ in dl:
        ok = torch.isfinite(r)
        if not ok.any():
            continue
        ids, mask, y = ids[ok].to(device), mask[ok].to(device), y[ok].to(device)
        yh.extend(rm(ids, mask, y, pop).cpu().numpy().tolist())
        yt.extend(r[ok].cpu().numpy().tolist())
    yt, yh = np.array(yt), np.array(yh)
    if len(yt) == 0:
        return {'n': 0}
    return {'n': int(len(yt)), 'mae': float(np.abs(yt - yh).mean()),
            'rmse': float(np.sqrt(np.mean((yt-yh)**2))),
            'constant_mean_baseline_mae': None,  # filled with train mean below
            'pearson_r': float(np.corrcoef(yt, yh)[0, 1]) if np.std(yh) > 1e-7 and np.std(yt) > 1e-7 else None}


@torch.no_grad()
def evaluate(model, rm, rows, args, device, n, pop, return_query=False):
    """Evaluate chronologically held-out queries; optionally expose paired query metrics."""
    model.eval()
    dl = DataLoader(SeqDataset(rows), batch_size=args.batch, shuffle=False,
                    collate_fn=lambda b: collate(b, n, n + 1, args.ctx))
    k = min(10, n)
    hit, ndcg, recommended, exposure, reward_scores = 0, 0.0, set(), Counter(), []
    avg_pop = []
    qrows = []
    for ids, mask, lengths, y, _, seen, userids in dl:
        ids, mask, lengths, y = (x.to(device) for x in (ids, mask, lengths, y))
        scores = mask_seen(last_logits(model, ids, mask, lengths), seen, n)
        top = torch.topk(scores, k, -1).indices
        match = (top == y[:, None])
        found = match.any(1)
        discount = found.float() / torch.log2(match.float().argmax(1).float() + 2)
        hit += int(found.sum())
        ndcg += float(discount.sum())
        top_pop = pop[top].mean(1)
        avg_pop.extend(top_pop.cpu().numpy().tolist())
        for items in top.cpu().tolist():
            recommended.update(items)
            exposure.update(items)
        batch_rewards = rm(ids, mask, top[:, 0], pop).cpu().numpy().tolist()
        reward_scores.extend(batch_rewards)
        if return_query:
            for u, h, d, reward, popularity in zip(
                    userids, found.cpu().numpy(), discount.cpu().numpy(),
                    batch_rewards, top_pop.cpu().numpy()):
                qrows.append({'user_id': int(u), 'hit@10': float(h), 'ndcg@10': float(d),
                              'predicted_reward_top1': float(reward),
                              'mean_train_popularity@10': float(popularity)})
    total = len(rows)
    result = {'queries': total, 'recall@10': round(hit / total, 5),
              'ndcg@10': round(ndcg / total, 5),
              'catalog_coverage@10': round(len(recommended)/n, 5),
              'top1_share_of_top10_exposures': round(max(exposure.values()) / max(1, total*k), 5),
              'mean_train_popularity@10': round(float(np.mean(avg_pop)), 5),
              'predicted_reward_top1': round(float(np.mean(reward_scores)), 5)}
    return (result, qrows) if return_query else result


def paired_cluster_bootstrap(base_rows, compare_rows, seed, repetitions=2000):
    """Paired bootstrap by user, keeping repeated queries from the same user together.

    This is descriptive uncertainty on this fixed held-out user/sample selection,
    not a causal or out-of-sample guarantee.
    """
    assert len(base_rows) == len(compare_rows)
    assert all(b['user_id'] == c['user_id'] for b, c in zip(base_rows, compare_rows))
    users, inv = np.unique([r['user_id'] for r in base_rows], return_inverse=True)
    sizes = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(users), size=(repetitions, len(users)))
    denom = sizes[draw].sum(axis=1)
    ci = {'test_users': int(len(users)), 'bootstrap_repetitions': repetitions}
    for field in ('hit@10', 'ndcg@10', 'predicted_reward_top1', 'mean_train_popularity@10'):
        values = np.array([c[field]-b[field] for b,c in zip(base_rows,compare_rows)])
        group_sums = np.bincount(inv, weights=values, minlength=len(users))
        distribution = group_sums[draw].sum(axis=1) / denom
        ci[field] = {'delta': round(float(values.mean()), 6),
                     'ci95_lower': round(float(np.quantile(distribution, .025)), 6),
                     'ci95_upper': round(float(np.quantile(distribution, .975)), 6)}
    return ci


@torch.no_grad()
def rollout(model, rm, rows, args, device, n, pop):
    """Hypothetical, non-user-confirmed 5-step greedy rollout."""
    model.eval()
    traces = []
    sample = rows[:min(args.rollout_users, len(rows))]
    for history, _, _, uid in sample:
        h = history.copy()
        results = []
        for step in range(5):
            ids, mask, lens, _, _, seen, _ = collate([(h, 0, 0.0, uid)], n, n+1, args.ctx)
            ids, mask, lens = ids.to(device), mask.to(device), lens.to(device)
            logits = mask_seen(last_logits(model, ids, mask, lens), seen, n)
            candidate = int(logits.argmax(dim=-1).item())
            expected_reward = float(rm(ids, mask, torch.tensor([candidate], device=device), pop).item())
            results.append(expected_reward)
            h.append(candidate)  # Simulated click: a strong, unrealistic assumption!
        traces.append(results)
    return [round(float(x), 5) for x in np.mean(traces, axis=0)] if traces else []
