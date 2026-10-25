"""Neural architectures and candidate masking."""

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

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
