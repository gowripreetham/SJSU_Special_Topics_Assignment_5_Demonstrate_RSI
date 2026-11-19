"""Method 1 — AIDE-style solution-tree search.

The idea (from AIDE, "AI-Driven Exploration in the Space of Code"): treat improvement as a search over a TREE of
complete solutions. Each node is a full program with its measured score. Three operators grow the tree:
    draft    write a new solution from scratch          (several roots → diversity)
    debug    repair a node that crashed                 (bounded depth, so a hopeless branch is abandoned)
    improve  make one atomic change to a working node   (normally the best node so far: greedy, with a little exploration)
Every node is fully evaluated, and the whole journal is kept — failures included.

Laptop adaptation: the "solutions" are one-line dispatch rules instead of ML scripts, and the default proposer is a
grammar-based editor instead of an LLM. The search policy is the part being demonstrated.
"""
from __future__ import annotations
import random

from .. import program
from ..env import BudgetExhausted, Env
from .common import START, Node, best_of, default_proposer, evaluate_node


def aide_tree(env: Env, rng: random.Random, proposer=None, n_drafts: int = 4, p_debug: float = 0.5, max_debug_depth: int = 2, explore: float = 0.15) -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []

    def debug_depth(n: Node) -> int:
        depth = 0
        while n.op == "debug" and n.parent is not None:
            depth, n = depth + 1, nodes[n.parent]
        return depth

    try:
        evaluate_node(env, nodes, START, "draft", None)                       # the first draft is the shared starting rule
        while True:
            drafts = [n for n in nodes if n.op == "draft"]
            children = {n.parent for n in nodes}
            buggy_leaves = [n for n in nodes if not n.result.ok and n.id not in children and debug_depth(n) < max_debug_depth and not n.result.error.startswith("rejected")]
            good = [n for n in nodes if n.result.ok]
            if len(drafts) < n_drafts:
                evaluate_node(env, nodes, proposer.edit("draft", START, rng), "draft", None)
            elif buggy_leaves and rng.random() < p_debug:
                target = rng.choice(buggy_leaves)
                evaluate_node(env, nodes, proposer.edit("debug", target.rule, rng, hint=f"It crashed with: {target.result.error}"), "debug", target.id)
            else:
                parent = rng.choice(good) if rng.random() < explore else best_of(good)
                op = rng.choice(program.IMPROVE_OPS)
                evaluate_node(env, nodes, proposer.edit(op, parent.rule, rng), op, parent.id)
    except BudgetExhausted:
        return nodes
