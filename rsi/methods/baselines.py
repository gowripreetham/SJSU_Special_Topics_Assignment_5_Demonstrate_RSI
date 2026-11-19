"""Two reference points every RSI method must beat at the same budget.

random_search   no inheritance: edit a random earlier rule (or draft a new one), keep the best ever seen
naive_rsi       first-order hill climbing: edit the incumbent, accept the child only if it scores better
"""
from __future__ import annotations
import random

from .. import program
from ..env import BudgetExhausted, Env
from .common import START, Node, best_of, default_proposer, evaluate_node


def random_search(env: Env, rng: random.Random, proposer=None) -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []
    try:
        evaluate_node(env, nodes, START, "start", None)
        while True:
            if rng.random() < 0.3:
                evaluate_node(env, nodes, proposer.edit("draft", START, rng), "draft", None)
            else:
                parent = rng.choice([n for n in nodes if n.result.ok])
                op = rng.choice(program.IMPROVE_OPS)
                evaluate_node(env, nodes, proposer.edit(op, parent.rule, rng), op, parent.id)
    except BudgetExhausted:
        return nodes


def naive_rsi(env: Env, rng: random.Random, proposer=None) -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []
    try:
        incumbent = evaluate_node(env, nodes, START, "start", None)
        while True:
            op = rng.choice(program.IMPROVE_OPS)
            child = evaluate_node(env, nodes, proposer.edit(op, incumbent.rule, rng), op, incumbent.id)
            if child.score < incumbent.score:
                incumbent = child
    except BudgetExhausted:
        return nodes
