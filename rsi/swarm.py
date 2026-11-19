"""Group 03 — scaling RSI with several agents (laptop adaptation of the coordination idea in the course's AReaL-RSI notes).

independent   N workers, each with budget/N and its own private history. Nothing is shared; the best result wins.
coordinated   the same N workers and the same total budget, plus a LEADER that reviews after every campaign:
                · orientation   each worker gets a different family of edits, so they do not duplicate each other
                · supported findings   the best rule so far is published; a worker that is clearly behind restarts from it
                · unresolved problems  rules that crashed are shared, so no worker pays to rediscover the same failure
Each worker is a first-order hill climber; only the coordination differs.
"""
from __future__ import annotations
import random

from . import program
from .env import BudgetExhausted, Env
from .methods.baselines import naive_rsi
from .methods.common import START, Node, best_of, default_proposer, evaluate_node

ORIENTATIONS = [["tweak_const", "add_term"], ["mul_term", "swap_var", "drop_term"], ["wrap_urgency", "add_term", "drop_term"]]


def independent(budget: int, seed: int, workers: int = 3) -> tuple[list[Node], int]:
    best: list[Node] = []
    spent = 0
    for k in range(workers):
        env = Env(budget=budget // workers)
        nodes = naive_rsi(env, random.Random(seed * 100 + k))
        spent += env.spent
        best.append(best_of(nodes))
    return best, spent


def coordinated(budget: int, seed: int, workers: int = 3, campaign: int = 6) -> tuple[list[Node], int, dict]:
    env, proposer = Env(budget=budget), default_proposer()
    rngs = [random.Random(seed * 100 + k) for k in range(workers)]
    nodes: list[Node] = []
    failed: set[str] = set()                                    # unresolved problems, shared
    log = {"adoptions": 0, "skipped_known_failures": 0, "campaigns": 0}
    try:
        start = evaluate_node(env, nodes, START, "start", None)
        incumbents = [start] * workers
        while True:
            for k in range(workers):                             # one campaign: every worker takes `campaign` steps in its orientation
                for _ in range(campaign):
                    op = rngs[k].choice(ORIENTATIONS[k % len(ORIENTATIONS)])
                    rule = proposer.edit(op, incumbents[k].rule, rngs[k])
                    if rule in failed:
                        log["skipped_known_failures"] += 1
                        continue
                    child = evaluate_node(env, nodes, rule, op, incumbents[k].id, worker=k)
                    if not child.result.ok:
                        failed.add(rule)
                    elif child.score < incumbents[k].score:
                        incumbents[k] = child
            log["campaigns"] += 1
            leader = best_of(incumbents)                         # leader review: publish the supported finding
            for k in range(workers):
                if incumbents[k].score > leader.score * 1.02:
                    incumbents[k] = leader
                    log["adoptions"] += 1
    except BudgetExhausted:
        return [best_of(nodes)], env.spent, log
