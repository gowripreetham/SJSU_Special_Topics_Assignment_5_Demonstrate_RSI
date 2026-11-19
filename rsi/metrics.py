"""Group 05 — how improvement is measured here.

RSI-Index      fraction of the gap between the STARTING rule and a REFERENCE that a run closed, on held-out data:
                   (start − achieved) / (start − reference)
               0 = no better than where it started; 1 = as good as the reference; >1 = better than the reference.
               Reference = ATC with its look-ahead constant tuned on the train split — a strong hand-designed rule.
               (The course notes describe RSI-Index as "mean fraction of remaining performance gap closed"; this is that
               idea with an explicit, attainable reference instead of a perfect score, which no rule can reach here.)
generalisation gap   test − train relative cost of the selected rule: how much of the gain was fitted to the train shifts
thrash rate          anomalies per 100 steps in a first-order trace (see methods/so_rsi.py)
"""
from __future__ import annotations
import random
import statistics

from . import problem
from .env import Env, held_out


def tuned_atc() -> tuple[str, float]:
    """Grid-search ATC's look-ahead k on TRAIN only. Returns (rule, train score)."""
    env = Env(budget=10**9)
    best = min(((f"w / p * exp(-max(0, slack) / ({k} * p_avg))", env.evaluate(f"w / p * exp(-max(0, slack) / ({k} * p_avg))").score) for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)), key=lambda x: x[1])
    return best


def rsi_index(achieved: float, start: float, reference: float) -> float:
    return (start - achieved) / (start - reference)


def references() -> dict:
    rule, _ = tuned_atc()
    out = {name: held_out(src) for name, src in problem.CLASSICAL.items()}
    out["ATC (k tuned on train)"] = held_out(rule) | {"rule": rule}
    return out


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def paired_bootstrap(a: list[float], b: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a − b) over paired seeds with a 95% bootstrap interval. Negative = a has lower cost."""
    rng, d = random.Random(seed), [x - y for x, y in zip(a, b)]
    means = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return statistics.mean(d), means[int(0.025 * n)], means[int(0.975 * n)]
