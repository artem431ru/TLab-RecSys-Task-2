"""Command-line configuration for the experiment."""

import argparse


def parse_args(argv=None):
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
    return parser.parse_args(argv)
