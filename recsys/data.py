"""Time-based KION dataset loading and batching."""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad, bos, ctx):
    histories, targets, rewards, userids = zip(*batch)
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
    return ids, mask, lengths, y, r, seen, tuple(userids)


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
                result[period].append((hist.copy(), token, reward, int(uid)))
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
