"""Run the KION GPT-2 + reward model + REINFORCE experiment.

Example:
    python run_experiment.py --data ../recsys_research/data/interactions.csv
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


def main():
    args = parse_args()
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
    train_rew = np.array([r for _, _, r, _ in rows['train'] if math.isfinite(r)])
    train_mean = float(train_rew.mean())
    for p in ('val', 'test'):
        vals = [r for _, _, r, _ in rows[p] if math.isfinite(r)]
        reward_metrics[p]['constant_mean_baseline_mae'] = float(np.mean(np.abs(np.array(vals)-train_mean))) if vals else None
    metrics = {'val': {}, 'test': {}}
    test_query_metrics = {}
    policies = {'GPT-2 SFT': base}
    for period in ('val', 'test'):
        if period == 'test':
            metrics[period]['GPT-2 SFT'], test_query_metrics['GPT-2 SFT'] = evaluate(
                base, rm, rows[period], args, device, n, pop, return_query=True)
        else:
            metrics[period]['GPT-2 SFT'] = evaluate(base, rm, rows[period], args, device, n, pop)
    traces = {}
    for name, mode in [('RL: raw watched_pct', 'raw'), ('RL: novelty penalty', 'novelty')]:
        print(f'RL mode={mode}...', flush=True)
        policy, trace = train_rl(base, rm, rows['train'], args, device, n, pop, mode)
        policies[name] = policy
        traces[mode] = trace
        for period in ('val', 'test'):
            if period == 'test':
                metrics[period][name], test_query_metrics[name] = evaluate(
                    policy, rm, rows[period], args, device, n, pop, return_query=True)
            else:
                metrics[period][name] = evaluate(policy, rm, rows[period], args, device, n, pop)
    confidence = {name: paired_cluster_bootstrap(test_query_metrics['GPT-2 SFT'],
                                                   test_query_metrics[name], args.seed)
                  for name in ('RL: raw watched_pct', 'RL: novelty penalty')}
    print('95% ДИ bootstrap по пользователям:', json.dumps(confidence, ensure_ascii=False), flush=True)
    pd.DataFrame([{'model': name, 'query_id': i, **rec}
                  for name, items in test_query_metrics.items()
                  for i, rec in enumerate(items)]).to_csv(out/'test_query_metrics.csv', index=False)
    (out/'test_bootstrap.json').write_text(json.dumps(confidence, ensure_ascii=False, indent=2), encoding='utf-8')
    rollouts = {name: rollout(policy, rm, rows['test'], args, device, n, pop)
                for name, policy in policies.items()}
    result = {'coverage': coverage, 'reward_model': reward_metrics,
              'evaluation': metrics, 'five_step_rollout_predicted_reward': rollouts,
              'confidence_intervals': confidence,
              'config': vars(args)}
    (out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'rl_trace.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'split': p, 'model': model, **m} for p, v in metrics.items() for model, m in v.items()]).to_csv(
        out/'comparison.csv', index=False)
    report = render_report(args, coverage, reward_metrics, metrics, rollouts, traces, time.time() - start, confidence)
    (out/'REPORT.md').write_text(report, encoding='utf-8')
    print('Отчёт сохранён в:', (out/'REPORT.md').resolve(), flush=True)
    print(pd.read_csv(out/'comparison.csv').to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
