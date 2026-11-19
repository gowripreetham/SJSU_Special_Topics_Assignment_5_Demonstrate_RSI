"""Method 2 — Budget RSI: score code edits BEFORE running them.

The idea (as summarised in the course's notes on "Recursive Self-Improvement on a Budget", arXiv 2610.29121 — I worked
from that summary, not from the paper): full evaluation is the expensive step, so do not spend it on every proposed edit.
    1. pick a parent by Thompson sampling over the evaluated nodes (Beta posterior on how many instances each one wins)
    2. propose K edits of that parent — proposing is cheap, nothing is run
    3. score each edit with a LINEAR READOUT  ŷ = wᵀ·[1, parent score, 8 edit features]  fitted on the edits already labelled
    4. fully evaluate only the top-scored edit; its true score becomes a new label and the readout is refitted
Until `n_fit` labels exist the choice among the K edits is random (that is how the first labels are collected).

readout="filter"  is my extension (lab 10.05): the readout only removes the predicted-worst half; the pick among the rest is random.
readout="parent"  is the ablation: the features are dropped, so all K siblings tie and the pick is random.
readout="none"    with K=1 degenerates to Thompson-sampled hill climbing.

Laptop adaptation: the features describe one-line rules (growth, variables introduced/removed, due-date awareness, …)
rather than multi-file diffs; labels are relative cost on the train split.
"""
from __future__ import annotations
import random

import numpy as np

from .. import program
from ..env import BudgetExhausted, Env
from .common import START, Node, default_proposer, evaluate_node

CLIP = 1.5      # labels are clipped so one crash (score 9.99) cannot dominate the fit


def features(parent: Node, child_rule: str, op: str, readout: str) -> list[float]:
    base = [1.0, parent.score]
    return base if readout == "parent" else base + program.edit_features(parent.rule, child_rule, op)   # "full" and "filter" use all features


def fit_readout(X: list[list[float]], y: list[float], ridge: float = 1.0) -> np.ndarray:
    A, b = np.asarray(X, float), np.minimum(np.asarray(y, float), CLIP)
    return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)


def thompson_parent(nodes: list[Node], rng: random.Random) -> Node:
    """Sample a plausible win-rate for every working node and take the argmax: good nodes are chosen often, uncertain ones sometimes."""
    good = [n for n in nodes if n.result.ok]
    ref = nodes[0].result.per_instance                         # the starting rule's cost on each instance
    def draw(n: Node) -> float:
        wins = sum(c < r for c, r in zip(n.result.per_instance, ref))
        return rng.betavariate(1 + wins, 1 + len(ref) - wins) - 0.5 * n.score   # tie-break toward lower cost
    return max(good, key=draw)


def budget_rsi(env: Env, rng: random.Random, proposer=None, k: int = 8, n_fit: int = 12, readout: str = "full") -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []
    X: list[list[float]] = []
    y: list[float] = []
    try:
        evaluate_node(env, nodes, START, "start", None)
        while True:
            parent = thompson_parent(nodes, rng)
            cands = []
            for _ in range(k):
                op = rng.choice(program.IMPROVE_OPS + ["draft"])
                rule = proposer.edit(op, parent.rule, rng)
                if rule != parent.rule and all(rule != n.rule for n in nodes) and all(rule != c[1] for c in cands):
                    cands.append((op, rule))
            if not cands:
                env.idle()                         # nothing new to consider: counts toward the stall limit
                continue
            feats = [features(parent, rule, op, readout) for op, rule in cands]
            if len(y) >= n_fit and readout != "none":
                w = fit_readout(X, y)
                pred = [float(np.dot(w, f)) for f in feats]
                order = sorted(range(len(cands)), key=lambda i: (pred[i], rng.random()))
                if readout == "filter":            # extension (lab 10.05): use the readout only to discard the predicted-worst half
                    pick = rng.choice(order[:max(1, (len(order) + 1) // 2)])
                else:
                    pick = order[0]
                predicted = pred[pick]
            else:
                pick, predicted = rng.randrange(len(cands)), None
            op, rule = cands[pick]
            child = evaluate_node(env, nodes, rule, op, parent.id, predicted=predicted, considered=len(cands))
            X.append(feats[pick])
            y.append(child.score)
    except BudgetExhausted:
        return nodes
