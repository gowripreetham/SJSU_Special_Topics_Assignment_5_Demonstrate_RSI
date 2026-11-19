"""Helpers shared by the lab scripts."""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.env import Env, held_out  # noqa: E402,F401
from rsi.methods.common import best_of  # noqa: E402,F401

BUDGET = 2400
DEV_SEEDS = range(5)          # labs use development seeds; the reported study (rsi/experiments.py) uses seeds 100+


def header(lab: str, title: str, question: str) -> None:
    print(f"\n=== Lab {lab} — {title} ===\nQuestion: {question}\n")


def results() -> dict | None:
    p = ROOT / "results/results.json"
    return json.loads(p.read_text()) if p.exists() else None


def run(fn, seed: int, budget: int = BUDGET, **kw):
    env = Env(budget=budget)
    nodes = fn(env, random.Random(seed), **kw)
    return env, nodes


def table(rows: list[list], head: list[str]) -> None:
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


mean = statistics.mean
