"""Entry point for GPT-2 → reward model → KL-regularized REINFORCE experiments.

Generated reports and metrics are written exclusively to results/.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.config import parse_args
from recsys.data import SeqDataset, collate, load_data
from recsys.evaluation import evaluate, evaluate_rm, paired_cluster_bootstrap, rollout
from recsys.models import RewardModel, build_lm
from recsys.report import render_report
from recsys.training import train_reward, train_rl, train_sft
from recsys.utils import seed_all


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    start = time.time()
    seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto' else args.device
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, n, counts, coverage = load_data(args)
    pop = torch.tensor(
        np.log1p(counts) / max(1.0, np.log1p(counts.max())),
        dtype=torch.float32, device=device,
    )

    def make_loader(samples, shuffle=False):
        return DataLoader(
            SeqDataset(samples), batch_size=args.batch, shuffle=shuffle,
            collate_fn=lambda batch: collate(batch, n, n + 1, args.ctx),
        )

    train_loader = make_loader(rows['train'], shuffle=True)

    print('Device:', device, 'Обучаем GPT-2...', flush=True)
    baseline = build_lm(n, args.ctx).to(device)
    train_sft(baseline, train_loader, args, device)
    baseline.eval()

    print('Обучаем Reward Model...', flush=True)
    reward_model = RewardModel(n).to(device)
    train_reward(reward_model, train_loader, args, device, pop)
    reward_metrics = {
        split: evaluate_rm(reward_model, rows[split], args, device, n, pop)
        for split in ('val', 'test')
    }
    train_rewards = np.array([r for _, _, r, _ in rows['train'] if math.isfinite(r)])
    train_mean = float(train_rewards.mean())
    for split in ('val', 'test'):
        actual = np.array([r for _, _, r, _ in rows[split] if math.isfinite(r)])
        reward_metrics[split]['constant_mean_baseline_mae'] = (
            float(np.mean(np.abs(actual - train_mean))) if actual.size else None
        )

    policies = {'GPT-2 SFT': baseline}
    traces = {}
    for label, mode in (
        ('RL: raw watched_pct', 'raw'),
        ('RL: novelty penalty', 'novelty'),
    ):
        print(f'RL mode={mode}...', flush=True)
        policy, trace = train_rl(
            baseline, reward_model, rows['train'], args, device, n, pop, mode,
        )
        policies[label] = policy
        traces[mode] = trace

    metrics = {'val': {}, 'test': {}}
    query_rows = {}
    for split in ('val', 'test'):
        for label, policy in policies.items():
            if split == 'test':
                summary, per_query = evaluate(
                    policy, reward_model, rows[split], args, device, n, pop,
                    return_query=True,
                )
                query_rows[label] = per_query
            else:
                summary = evaluate(policy, reward_model, rows[split], args, device, n, pop)
            metrics[split][label] = summary

    baseline_rows = query_rows['GPT-2 SFT']
    confidence = {
        label: paired_cluster_bootstrap(
            baseline_rows, query_rows[label], seed=args.seed,
        )
        for label in policies if label != 'GPT-2 SFT'
    }

    rollouts = {
        label: rollout(policy, reward_model, rows['test'], args, device, n, pop)
        for label, policy in policies.items()
    }
    result = {
        'coverage': coverage,
        'reward_model': reward_metrics,
        'evaluation': metrics,
        'five_step_rollout_predicted_reward': rollouts,
        'config': vars(args),
    }
    write_json(out / 'metrics.json', result)
    write_json(out / 'rl_trace.json', traces)
    write_json(out / 'test_bootstrap.json', confidence)

    pd.DataFrame([
        {'split': split, 'model': label, **values}
        for split, group in metrics.items()
        for label, values in group.items()
    ]).to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame([
        {'query_index': index, 'model': label, **values}
        for label, samples in query_rows.items()
        for index, values in enumerate(samples)
    ]).to_csv(out / 'test_query_metrics.csv', index=False)

    report = render_report(
        args, coverage, reward_metrics, metrics, rollouts, traces,
        time.time() - start, confidence,
    )
    (out / 'REPORT.md').write_text(report, encoding='utf-8')

    print('\nЭксперимент завершён. Результаты:', out.resolve(), flush=True)
    print(pd.read_csv(out / 'comparison.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
