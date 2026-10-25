"""Lightweight sanity checks; run with `python -m pytest -q`."""

import numpy as np
import torch

from recsys.data import collate
from recsys.evaluation import paired_cluster_bootstrap


def test_batch_shapes_and_history_mask():
    batch = [([1, 2], 3, 0.7, 11), ([2], 4, 0.3, 12)]
    ids, mask, lengths, y, reward, seen, users = collate(batch, pad=10, bos=11, ctx=5)
    assert ids.shape == (2, 3)
    assert lengths.tolist() == [3, 2]
    assert y.tolist() == [3, 4]
    assert seen == [{1, 2}, {2}]
    assert users == (11, 12)
    assert mask.shape == ids.shape


def test_bootstrap_returns_exact_delta_for_constant_shift():
    baseline = [{'user_id': u, 'hit@10': 0.0, 'ndcg@10': 0.0,
                 'predicted_reward_top1': 0.4, 'mean_train_popularity@10': 0.4}
                for u in [1, 1, 2, 3]]
    changed = [{**r, 'predicted_reward_top1': 0.6} for r in baseline]
    ci = paired_cluster_bootstrap(baseline, changed, seed=42, repetitions=100)
    assert ci['test_users'] == 3
    assert np.isclose(ci['predicted_reward_top1']['delta'], 0.2)
    assert np.isclose(ci['predicted_reward_top1']['ci95_lower'], 0.2)
