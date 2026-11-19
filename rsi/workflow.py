"""Group 04 — adaptive workflows (laptop adaptations of the RSIFlow and Self-HPO ideas in the course notes).

adaptive_flow   The improvement loop is a workflow with several kinds of step. A small META-AGENT decides which step to run
                next from the loop's own execution feedback:
                    · an operator bandit (UCB1): reward = relative improvement an operator produced per evaluation
                    · bottleneck rules: many crashes → route to "debug"; long stagnation → route to "draft" a restart candidate
                So the workflow adapts as the bottleneck moves from "wrong structure" to "wrong constants".

self_hpo        The improver tunes ITSELF: it spends a slice of the budget on short trial runs of several configurations of the
                tree search, keeps the best half (successive halving), and gives the remaining budget to the winner.
                The object being improved is the search procedure — one level up from the dispatch rule.
"""
from __future__ import annotations
import math
import random
from collections import Counter

from . import program
from .env import BudgetExhausted, Env
from .methods.aide_tree import aide_tree
from .methods.common import START, Node, best_of, default_proposer, evaluate_node


def adaptive_flow(env: Env, rng: random.Random, proposer=None, stagnation_limit: int = 14) -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []
    pulls, reward, routes = Counter(), Counter(), Counter()
    try:
        incumbent = evaluate_node(env, nodes, START, "start", None)
        stale, t = 0, 0
        while True:
            t += 1
            recent = nodes[-8:]
            crashed = [n for n in recent if not n.result.ok and not n.result.error.startswith("rejected")]
            if len(crashed) >= 3 and not crashed[-1].meta.get("debugged"):          # bottleneck: edits keep crashing
                routes["debug"] += 1
                crashed[-1].meta["debugged"] = True
                child = evaluate_node(env, nodes, proposer.edit("debug", crashed[-1].rule, rng), "debug", crashed[-1].id)
            elif stale >= stagnation_limit:                                           # bottleneck: local search is exhausted
                routes["draft"] += 1
                stale = 0
                child = evaluate_node(env, nodes, proposer.edit("draft", START, rng), "draft", None)
            else:                                                                     # normal step: the bandit picks the operator
                routes["improve"] += 1
                ops = program.IMPROVE_OPS
                op = max(ops, key=lambda o: math.inf if pulls[o] == 0 else reward[o] / pulls[o] + 0.05 * math.sqrt(2 * math.log(t) / pulls[o]))
                child = evaluate_node(env, nodes, proposer.edit(op, incumbent.rule, rng), op, incumbent.id)
                pulls[op] += 1
                reward[op] += max(0.0, incumbent.score - child.score) / incumbent.score if child.result.ok else 0.0
            if child.score < incumbent.score:
                incumbent, stale = child, 0
            else:
                stale += 1
    except BudgetExhausted:
        pass
    if nodes:
        nodes[0].meta |= {"routes": dict(routes), "pulls": dict(pulls), "mean_reward": {o: round(reward[o] / pulls[o], 4) for o in pulls}}
    return nodes


CONFIGS = [{"n_drafts": 2, "explore": 0.0, "p_debug": 0.5}, {"n_drafts": 4, "explore": 0.15, "p_debug": 0.5}, {"n_drafts": 8, "explore": 0.15, "p_debug": 0.5},
           {"n_drafts": 4, "explore": 0.4, "p_debug": 0.2}, {"n_drafts": 2, "explore": 0.3, "p_debug": 0.8}, {"n_drafts": 6, "explore": 0.05, "p_debug": 0.5}]


def self_hpo(budget: int, seed: int, tune_fraction: float = 0.3) -> tuple[Node, int, dict]:
    """Returns (best node, shifts spent, record of the tuning). Total spend never exceeds `budget`."""
    tune = int(budget * tune_fraction)
    alive, record, spent, best_nodes = list(range(len(CONFIGS))), [], 0, []
    rounds = 2
    for r in range(rounds):                                        # successive halving: short runs, keep the better half
        per = tune // rounds // len(alive)
        scores = {}
        for c in alive:
            env = Env(budget=per)
            nodes = aide_tree(env, random.Random(seed * 1000 + r * 10 + c), **CONFIGS[c])
            spent += env.spent
            scores[c] = best_of(nodes).score
            best_nodes.append(best_of(nodes))
        record.append({"round": r, "shifts_each": per, "scores": {str(c): round(s, 4) for c, s in scores.items()}})
        alive = sorted(alive, key=scores.get)[:max(1, len(alive) // 2)]
    winner = alive[0]
    env = Env(budget=budget - spent)
    nodes = aide_tree(env, random.Random(seed * 1000 + 999), **CONFIGS[winner])
    best_nodes.append(best_of(nodes))
    return best_of(best_nodes), spent + env.spent, {"winner": CONFIGS[winner], "rounds": record}
