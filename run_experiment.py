"""KION GPT-2 next-item baseline, watched-pct reward model, KL-constrained REINFORCE.

CPU-friendly research pipeline. The pipeline deliberately DOES NOT use test-period events
for vocabulary, fitting, reward optimization or hyperparameter selection.
"""
import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad, bos, ctx):
    histories, targets, rewards = zip(*batch)
    tokens = [[bos] + h[-(ctx - 1):] for h in histories]
    lengths = torch.tensor([len(s) for s in tokens], dtype=torch.long)
    ids = torch.full((len(tokens), max(lengths).item()), pad, dtype=torch.long)
    for i, tok in enumerate(tokens):
        ids[i, :len(tok)] = torch.tensor(tok)
    mask = (ids != pad).long()
    y = torch.tensor(targets, dtype=torch.long)
    r = torch.tensor(rewards, dtype=torch.float32)
    # The full observed history is used to mask previously consumed items, not just GPT context.
    seen = [set(h) for h in histories]
    return ids, mask, lengths, y, r, seen


def load_data(args):
    print('Читаем KION...', flush=True)
    df = pd.read_csv(args.data, usecols=['user_id', 'item_id', 'last_watch_dt', 'watched_pct'],
                     parse_dates=['last_watch_dt'])
    df['watched_pct'] = pd.to_numeric(df['watched_pct'], errors='coerce').clip(0, 100) / 100.0
    train_cut = pd.Timestamp('2021-07-26')
    test_cut = pd.Timestamp('2021-08-09')
    # Train-only catalog selection and train-only user selection. No future information in either.
    train_mask = df['last_watch_dt'] < train_cut
    popular = df.loc[train_mask, 'item_id'].value_counts().head(args.items)
    vocab = {int(v): i for i, v in enumerate(popular.index)}
    train_counts = df.loc[train_mask, 'user_id'].value_counts()
    eligible = train_counts[train_counts >= 3].index.to_numpy()
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.users:
        users = rng.choice(eligible, size=args.users, replace=False)
    else:
        users = eligible
    # Filtering the catalog removes tail events from histories; documented in report.
    raw_user_count = len(users)
    df = df.loc[df['user_id'].isin(users) & df['item_id'].isin(vocab)].copy()
    df['token'] = df['item_id'].map(vocab).astype('int64')
    df = df.sort_values(['user_id', 'last_watch_dt', 'item_id'], kind='mergesort')
    result = {'train': [], 'val': [], 'test': []}
    train_pop = np.zeros(len(vocab), dtype=np.float32)
    for uid, group in df.groupby('user_id', sort=False):
        hist = []
        for row in group.itertuples(index=False):
            token = int(row.token)
            period = ('train' if row.last_watch_dt < train_cut else
                      'val' if row.last_watch_dt < test_cut else 'test')
            if hist:
                reward = float(row.watched_pct) if pd.notna(row.watched_pct) else float('nan')
                result[period].append((hist.copy(), token, reward))
            if period == 'train':
                train_pop[token] += 1
            hist.append(token)
    rng = random.Random(args.seed)
    rng.shuffle(result['train'])
    if args.max_train > 0:
        result['train'] = result['train'][:args.max_train]
    for period in ('val', 'test'):
        rng.shuffle(result[period])
        if args.max_eval > 0:
            result[period] = result[period][:args.max_eval]
    counts = {k: len(v) for k, v in result.items()}
    if min(counts.values()) < 10:
        raise RuntimeError(f'Слишком мало примеров для оценки {counts}. Попробуй --users 15000 --items 1500.')
    coverage = dict(selected_users=raw_user_count, catalog_size=len(vocab),
                    rows_in_catalog_users=len(df), examples=counts,
                    note='All metrics on head-catalog, sampled users and filtered events only.')
    print('Подготовлено:', coverage, flush=True)
    return result, len(vocab), train_pop, coverage


class RewardModel(nn.Module):
    """P(watched fraction | actual observed history, candidate item), not causal uplift."""
    def __init__(self, n_items, dim=32):
        super().__init__()
        self.emb = nn.Embedding(n_items + 2, dim, padding_idx=n_items)
        self.net = nn.Sequential(nn.Linear(dim * 3 + 1, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, ids, mask, candidate, popularity):
        # BOS is ignored for embedding mean; only real item tokens are used.
        item_mask = (mask.bool() & (ids < popularity.numel())).float().unsqueeze(-1)
        hist = (self.emb(ids) * item_mask).sum(1) / item_mask.sum(1).clamp(min=1)
        target = self.emb(candidate)
        pop = popularity[candidate].view(-1, 1)
        x = torch.cat([hist, target, hist * target, pop], dim=-1)
        return self.net(x).flatten()


def build_lm(n, ctx):
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=n + 2, n_positions=ctx + 1, n_ctx=ctx + 1,
        n_embd=64, n_layer=2, n_head=4,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=n + 1, pad_token_id=n, use_cache=False))


def last_logits(lm, ids, mask, lengths):
    logits = lm(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return logits[torch.arange(len(ids), device=ids.device), lengths - 1]


def mask_seen(logits, seen, n_items):
    logits = logits.clone()
    logits[:, n_items:] = -1e9
    # Mask all consumed items. Ensures no repeats in candidates or RL actions.
    for i, s in enumerate(seen):
        if s:
            logits[i, list(s)] = -1e9
    return logits


def train_sft(model, dl, args, device):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    last = 0.0
    for epoch in range(args.epochs):
        weighted = 0
        count = 0
        for ids, mask, lengths, y, _, _ in dl:
            ids, mask, lengths, y = (x.to(device) for x in (ids, mask, lengths, y))
            loss = F.cross_entropy(last_logits(model, ids, mask, lengths), y)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            weighted += loss.item() * len(y)
            count += len(y)
        last = weighted / max(1, count)
        print(f'  SFT epoch {epoch + 1}/{args.epochs}: CE={last:.4f}', flush=True)
    return last


def train_reward(rm, dl, args, device, pop):
    optim = torch.optim.AdamW(rm.parameters(), lr=0.003)
    rm.train()
    for epoch in range(args.rm_epochs):
        total, count = 0.0, 0
        for ids, mask, _, y, reward, _ in dl:
            ok = torch.isfinite(reward)
            if not ok.any():
                continue
            ids, mask, y, reward = (x.to(device) for x in (ids[ok], mask[ok], y[ok], reward[ok]))
            pred = rm(ids, mask, y, pop)
            loss = F.mse_loss(pred, reward)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(y)
            count += len(y)
        print(f'  RM epoch {epoch+1}/{args.rm_epochs}: MSE={total/max(1,count):.4f}', flush=True)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def evaluate_rm(rm, rows, args, device, n, pop):
    dl = DataLoader(SeqDataset(rows), batch_size=args.batch, shuffle=False,
                    collate_fn=lambda b: collate(b, n, n + 1, args.ctx))
    yh, yt = [], []
    for ids, mask, _, y, r, _ in dl:
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


def train_rl(base, rm, rows, args, device, n, pop, mode):
    policy = copy.deepcopy(base).to(device)
    policy.requires_grad_(True)  # baseline is frozen after first RL run
    policy.train()
    ref = base
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ds = SeqDataset(rows)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate(b, n, n+1, args.ctx))
    it = iter(loader)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.rl_lr)
    trace = []
    for step in range(args.rl_steps):
        try:
            ids, mask, lengths, y, _, seen = next(it)
        except StopIteration:
            it = iter(loader)
            ids, mask, lengths, y, _, seen = next(it)
        ids, mask, lengths, y = (x.to(device) for x in (ids, mask, lengths, y))
        current_raw = last_logits(policy, ids, mask, lengths)
        current = mask_seen(current_raw, seen, n)
        with torch.no_grad():
            reference = mask_seen(last_logits(ref, ids, mask, lengths), seen, n)
        # Sample policy actions; observed y provides an independent supervised anchor.
        logpi = F.log_softmax(current, -1)
        pi = logpi.exp()
        action = torch.multinomial(pi.detach(), 1).squeeze(1)
        with torch.no_grad():
            score = rm(ids, mask, action, pop)
            if mode == 'novelty':
                score = score - args.novelty_coef * pop[action]
            advantage = (score - score.mean()) / score.std(unbiased=False).clamp(min=0.05)
        reinforce = -(advantage * logpi.gather(1, action[:, None]).squeeze(1)).mean()
        logref = F.log_softmax(reference, -1)
        kl = (pi * (logpi - logref)).sum(-1).mean()
        # The anchor is supervised on actual next events and prevents extreme drift.
        anchor = F.cross_entropy(current_raw, y)
        loss = reinforce + args.kl * kl + args.anchor * anchor
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        if (step + 1) % max(1, args.rl_steps // 4) == 0 or step == 0:
            point = {'step': step + 1, 'mean_training_reward': round(float(score.mean()), 4),
                     'kl': round(float(kl.detach()), 4), 'anchor_ce': round(float(anchor.detach()), 4)}
            trace.append(point)
            print(' ', mode, point, flush=True)
    policy.eval()
    return policy, trace


@torch.no_grad()
def evaluate(model, rm, rows, args, device, n, pop):
    model.eval()
    dl = DataLoader(SeqDataset(rows), batch_size=args.batch, shuffle=False,
                    collate_fn=lambda b: collate(b, n, n + 1, args.ctx))
    k = min(10, n)
    hit, ndcg, recommended, exposure, reward_scores = 0, 0.0, set(), Counter(), []
    for ids, mask, lengths, y, _, seen in dl:
        ids, mask, lengths, y = (x.to(device) for x in (ids, mask, lengths, y))
        scores = mask_seen(last_logits(model, ids, mask, lengths), seen, n)
        top = torch.topk(scores, k, -1).indices
        match = (top == y[:, None])
        found = match.any(1)
        hit += int(found.sum())
        pos = match.float().argmax(1).float()
        ndcg += float((found.float() / torch.log2(pos + 2)).sum())
        for items in top.cpu().tolist():
            recommended.update(items)
            exposure.update(items)
        reward_scores.extend(rm(ids, mask, top[:, 0], pop).cpu().numpy().tolist())
    total = len(rows)
    return {'queries': total, 'recall@10': round(hit / total, 5),
            'ndcg@10': round(ndcg / total, 5),
            'catalog_coverage@10': round(len(recommended)/n, 5),
            'top1_share_of_top10_exposures': round(max(exposure.values()) / max(1, total*k), 5),
            'predicted_reward_top1': round(float(np.mean(reward_scores)), 5)}


@torch.no_grad()
def rollout(model, rm, rows, args, device, n, pop):
    """Hypothetical, non-user-confirmed 5-step greedy rollout."""
    model.eval()
    traces = []
    sample = rows[:min(args.rollout_users, len(rows))]
    for history, _, _ in sample:
        h = history.copy()
        results = []
        for step in range(5):
            ids, mask, lens, _, _, seen = collate([(h, 0, 0.0)], n, n+1, args.ctx)
            ids, mask, lens = ids.to(device), mask.to(device), lens.to(device)
            logits = mask_seen(last_logits(model, ids, mask, lens), seen, n)
            candidate = int(logits.argmax(dim=-1).item())
            expected_reward = float(rm(ids, mask, torch.tensor([candidate], device=device), pop).item())
            results.append(expected_reward)
            h.append(candidate)  # Simulated click: a strong, unrealistic assumption!
        traces.append(results)
    return [round(float(x), 5) for x in np.mean(traces, axis=0)] if traces else []


def render_report(args, coverage, reward_metrics, metrics, rollouts, traces, runtime):
    def table(d):
        out = ['| Модель | Recall@10 | NDCG@10 | Покрытие каталога@10 | RM reward@top1 |',
               '|---|---:|---:|---:|---:|']
        for name, m in d.items():
            out.append(f'| {name} | {m["recall@10"]:.5f} | {m["ndcg@10"]:.5f} | '
                       f'{m["catalog_coverage@10"]:.5f} | {m["predicted_reward_top1"]:.5f} |')
        return '\n'.join(out)
    def rmrow(m):
        return f'MAE={m.get("mae", float("nan")):.4f}, RMSE={m.get("rmse", float("nan")):.4f}, n={m.get("n", 0)}'
    baseline = metrics['test']['GPT-2 SFT']
    changes = []
    for name in ['RL: raw watched_pct', 'RL: novelty penalty']:
        met = metrics['test'][name]
        changes.append(f'- **{name}:** ΔNDCG@10 = {met["ndcg@10"]-baseline["ndcg@10"]:+.5f}; '
                       f'ΔRecall@10 = {met["recall@10"]-baseline["recall@10"]:+.5f}; '
                       f'ΔRM reward = {met["predicted_reward_top1"]-baseline["predicted_reward_top1"]:+.5f}.')
    conclusions = ('Повышение модельной награды не доказывает рост удовлетворённости пользователей. '
                   'Выводы ограничены офлайн-наблюдениями и малой экспериментальной выборкой.')
    report = f'''# Отчёт: RL-дообучение GPT-2 для последовательных рекомендаций KION

## Постановка

Задача: сравнить next-item модель GPT-2 до/после RL, оценивая релевантность и прогнозируемую долю досмотра. Исследование проведено на **реальном KION** (не синтетических данных). В качестве токенов используются ID фильмов/сериалов.

## Данные и протокол

- Разбиение по `last_watch_dt`: train до **26.07.2021**, validation с 26.07 по 08.08 включительно, test с **09.08.2021**.
- Словарь из {coverage['catalog_size']} самых частых объектов **только train**. Выборка {coverage['selected_users']} пользователей с ≥3 train-событиями, случайная с фиксированным seed. Отбор пользователей не использует будущее.
- Train/validation/test примеров внутри выбранного каталога: {coverage['examples']['train']} / {coverage['examples']['val']} / {coverage['examples']['test']}.
- Истории учитывают лишь фактически наблюдавшиеся события, предшествующие целевому по дате. На test доступны известные реальные события validation как история, но обучение на них не проводится.
- Метрики рекомендаций вычислены **только на событиях из top-{coverage['catalog_size']} train-каталога**, а не на полном KION. Невидимые объекты каталога не могут быть угаданы, поэтому полученные числа нельзя напрямую сравнивать с full-catalog RecSys.
- `last_watch_dt` в KION представляет агрегированную дату последнего просмотра для пары user-item. Это приближённая хронология, а не полный журнал сессий; при равных датах реальный порядок неизвестен.
- Тестовая выборка не применялась при подборе словаря, обучении и RL. Все настройки зафиксированы заранее. Дубликаты user-item исключаются природой агрегированного датасета.

## Модели

1. **GPT-2 SFT**: GPT2LMHeadModel с 2 Transformer-блоками, 4 attention-heads, hidden=64, контекст до {args.ctx} токенов; задача предсказания следующего item ID по реальной истории. Обучение {args.epochs} эпох, AdamW, LR={args.lr}, CE.
2. **Reward model**: отдельная обучаемая нейросеть с embedding истории и объекта + MLP, sigmoid для прогноза `watched_pct/100` в [0,1]. Обучается {args.rm_epochs} эпох на **наблюдаемых** парах из train. Полезность **ненаблюдавшихся** объектов из этих данных не известна. Результаты валидации: {rmrow(reward_metrics['val'])}; теста: {rmrow(reward_metrics['test'])}. MAE константного среднего train на test: {reward_metrics['test'].get('constant_mean_baseline_mae')}.
3. **RL**: REINFORCE, действие = выбранный item; награда из frozen reward model, штраф KL(π||π_SFT) с коэффициентом {args.kl} и supervised-anchoring (CE) с коэффициентом {args.anchor}. {args.rl_steps} RL-шагов, LR={args.rl_lr}. Запрещены уже просмотренные объекты (mask seen). **Два независимых старта из одной SFT:** reward = predicted watched_pct; reward = predicted watched_pct − {args.novelty_coef} × train popularity, нормированная логарифмом.

## Релевантность и награда: Validation

{table(metrics['val'])}

## Релевантность и награда: Test

{table(metrics['test'])}

Recall@10 и NDCG@10 оцениваются на каждом следующем реально просмотренном объекте. При одном релевантном объекте на запрос Recall@10 совпадает с HitRate@10. `RM reward@top1` — средняя **предсказанная**, а не реальная оценка реакции на top-1 рекомендацию.

### Изменения по сравнению с SFT на Test

{chr(10).join(changes)}

Файлы `results/metrics.json`, `results/rl_trace.json` содержат полный набор результатов, включая coverage каталога и концентрацию популярности.

## Генерация 5 шагов

Здесь каждую собственную рекомендацию добавляем в историю как будто пользователь её выбрал. Это **симуляция, а не наблюдение**.

| Модель | RM reward шаги 1→5 |
|---|---|
{chr(10).join('| '+k+' | '+', '.join(f'{v:.3f}' for v in vals)+' |' for k,vals in rollouts.items())}

Условия оценки после первого шага изменяются: модель обучалась на настоящих историях, а получает собственные гипотетические действия. В отсутствие пользовательской модели и реальных ответов нельзя интерпретировать пятишаговые значения как доказанное долгосрочное улучшение.

## Компромисс и ограничения

{chr(10).join(changes)}

{conclusions} `watched_pct` может давать смещение к коротким объектам и просмотрам уже вовлечённых пользователей, а learned reward можно эксплуатировать в областях без достоверной разметки. Отсутствуют impression logs и корректные propensity scores. **Контрфактическую награду и retention офлайн подтвердить нельзя.** Для проверки возвращаемости требуются последовательные реальные пользовательские сессии с датами повторных визитов и A/B-тест.

Обратим внимание на концентрацию рекомендаций: покрытие каталога и доля самого популярного item в рекомендациях указаны в `metrics.json`. Низкое покрытие или растущая концентрация после RL свидетельствует о коллапсе разнообразия даже при росте predicted reward.

## Воспроизведение

```powershell
pip install -r requirements.txt
python run_experiment.py --data "../recsys_research/data/interactions.csv" --out results
```

Для запуска с более широкой выборкой: `python run_experiment.py --data <path/to/interactions.csv> --users 15000 --items 1000 --max-train 20000 --max-eval 1000 --epochs 2 --rl-steps 120`.

Случайность фиксирована `seed={args.seed}`. Устройство: автоматический выбор CUDA/CPU, общее время запуска: {runtime:.1f} сек. Получение KION: https://github.com/irsafilo/KION_DATASET. Экспериментальная среда и версии зависимостей — в `requirements.txt`; аппаратная конфигурация влияет на скорость и незначительно на численные результаты.
'''
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../recsys_research/data/interactions.csv')
    parser.add_argument('--out', default='results')
    parser.add_argument('--users', type=int, default=6000)
    parser.add_argument('--items', type=int, default=600)
    parser.add_argument('--max-train', type=int, default=5000)
    parser.add_argument('--max-eval', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--rm-epochs', type=int, default=2)
    parser.add_argument('--rl-steps', type=int, default=40)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--ctx', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--rl-lr', type=float, default=0.00005)
    parser.add_argument('--kl', type=float, default=0.1)
    parser.add_argument('--anchor', type=float, default=0.1)
    parser.add_argument('--novelty-coef', type=float, default=0.15)
    parser.add_argument('--rollout-users', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    start = time.time()
    seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    rows, n, counts, coverage = load_data(args)
    pop = torch.tensor(np.log1p(counts)/max(1., np.log1p(counts.max())), dtype=torch.float32, device=device)
    mkloader = lambda rs, shuf: DataLoader(SeqDataset(rs), batch_size=args.batch, shuffle=shuf,
                                         collate_fn=lambda b: collate(b, n, n + 1, args.ctx))
    trainloader = mkloader(rows['train'], True)
    print('Device:', device, 'Обучаем GPT-2...', flush=True)
    base = build_lm(n, args.ctx).to(device)
    train_sft(base, trainloader, args, device)
    base.eval()
    print('Обучаем Reward Model...', flush=True)
    rm = RewardModel(n).to(device)
    train_reward(rm, trainloader, args, device, pop)
    reward_metrics = {p: evaluate_rm(rm, rows[p], args, device, n, pop) for p in ('val', 'test')}
    train_rew = np.array([r for _, _, r in rows['train'] if math.isfinite(r)])
    train_mean = float(train_rew.mean())
    for p in ('val', 'test'):
        vals = [r for _, _, r in rows[p] if math.isfinite(r)]
        reward_metrics[p]['constant_mean_baseline_mae'] = float(np.mean(np.abs(np.array(vals)-train_mean))) if vals else None
    metrics = {'val': {}, 'test': {}}
    policies = {'GPT-2 SFT': base}
    for period in ('val', 'test'):
        metrics[period]['GPT-2 SFT'] = evaluate(base, rm, rows[period], args, device, n, pop)
    traces = {}
    for name, mode in [('RL: raw watched_pct', 'raw'), ('RL: novelty penalty', 'novelty')]:
        print(f'RL mode={mode}...', flush=True)
        policy, trace = train_rl(base, rm, rows['train'], args, device, n, pop, mode)
        policies[name] = policy
        traces[mode] = trace
        for period in ('val', 'test'):
            metrics[period][name] = evaluate(policy, rm, rows[period], args, device, n, pop)
    rollouts = {name: rollout(policy, rm, rows['test'], args, device, n, pop)
                for name, policy in policies.items()}
    result = {'coverage': coverage, 'reward_model': reward_metrics,
              'evaluation': metrics, 'five_step_rollout_predicted_reward': rollouts,
              'config': vars(args)}
    (out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'rl_trace.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'split': p, 'model': model, **m} for p, v in metrics.items() for model, m in v.items()]).to_csv(
        out/'comparison.csv', index=False)
    report = render_report(args, coverage, reward_metrics, metrics, rollouts, traces, time.time() - start)
    (out/'REPORT.md').write_text(report, encoding='utf-8')
    (Path(__file__).resolve().parent/'README.md').write_text(report, encoding='utf-8')
    print('\nГОТОВО! Отчёт записан в README.md, результаты: ', out.resolve(), flush=True)
    print(pd.read_csv(out/'comparison.csv').to_string(index=False), flush=True)
    print('Важное замечание: это небольшое офлайн-исследование, не проверка реальной долгосрочной полезности.', flush=True)


if __name__ == '__main__':
    main()
