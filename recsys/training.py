"""SFT, observed watch reward fitting, and KL-regularized REINFORCE."""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import SeqDataset, collate
from .models import last_logits, mask_seen

def train_sft(model, dl, args, device):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    last = 0.0
    for epoch in range(args.epochs):
        weighted = 0
        count = 0
        for ids, mask, lengths, y, _, _, _ in dl:
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
        for ids, mask, _, y, reward, _, _ in dl:
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
            ids, mask, lengths, y, _, seen, _ = next(it)
        except StopIteration:
            it = iter(loader)
            ids, mask, lengths, y, _, seen, _ = next(it)
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
