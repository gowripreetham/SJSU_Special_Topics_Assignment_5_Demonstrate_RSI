"""The problem every lab improves on: sequencing a hospital pharmacy's sterile-compounding queue.

Orders (IV bags, chemo doses, TPN) arrive over a shift. One clean-room hood prepares one order at a time, without
interruption. Each order has a preparation time p, a due time d (when the ward needs it), a clinical weight w (STAT > routine)
and an arrival time r. Whenever the hood is free, a DISPATCH RULE picks the next order: the waiting order with the highest
priority(order, now). The cost of a shift is total weighted tardiness  Σ w·max(0, finish − d)  — lower is better.

The dispatch rule is the object that gets improved: a one-line arithmetic expression such as  `w / p`.
All instances are SYNTHETIC (seeded random); nothing here is real hospital data.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# Variables a rule may read. Everything is known at decision time — no look-ahead.
VARIABLES = ("p", "w", "d", "t", "slack", "wait", "p_avg", "n")


@dataclass(frozen=True)
class Order:
    r: float   # arrival time (min)
    p: float   # preparation time (min)
    d: float   # due time (min)
    w: float   # clinical weight


def make_instance(seed: int, n: int = 40, load: float = 0.95, tightness: float = 2.2) -> tuple[Order, ...]:
    """One shift. `load` ≈ utilisation of the hood; `tightness` scales how much slack orders get (smaller = tighter)."""
    rng = random.Random(seed)
    kinds = [((3, 8), 1.0, 0.55), ((8, 20), 2.0, 0.30), ((20, 45), 4.0, 0.15)]   # (prep-time range, base weight, share): routine IV, antibiotic, chemo/TPN
    orders, t = [], 0.0
    mean_p = sum((lo + hi) / 2 * share for (lo, hi), _, share in kinds)
    for _ in range(n):
        t += rng.expovariate(load / mean_p)
        (lo, hi), base_w, _ = rng.choices(kinds, weights=[k[2] for k in kinds])[0]
        p = rng.uniform(lo, hi)
        stat = rng.random() < 0.15                                                 # STAT orders: urgent and heavily weighted
        w = base_w * (3.0 if stat else 1.0) * rng.uniform(0.8, 1.2)
        d = t + p * (1.2 if stat else tightness * rng.uniform(0.8, 2.5))
        orders.append(Order(round(t, 2), round(p, 2), round(d, 2), round(w, 2)))
    return tuple(orders)


def simulate(orders: tuple[Order, ...], priority, detail: list | None = None) -> float:
    """Run one shift under `priority(p, w, d, t, slack, wait, p_avg, n) -> float`. Returns total weighted tardiness.
    A rule that raises or returns a non-finite number makes the run fail (the caller records a crash).
    Pass a list as `detail` to also collect (order, weighted tardiness) pairs — used by diagnostic probes."""
    pending = sorted(range(len(orders)), key=lambda i: orders[i].r)
    queue: list[int] = []
    t = cost = 0.0
    k = 0
    while k < len(pending) or queue:
        while k < len(pending) and orders[pending[k]].r <= t:
            queue.append(pending[k]); k += 1
        if not queue:
            t = orders[pending[k]].r
            continue
        p_avg, n = sum(orders[i].p for i in queue) / len(queue), len(queue)
        best, best_score = queue[0], -math.inf
        for i in queue:
            o = orders[i]
            s = priority(o.p, o.w, o.d, t, o.d - t - o.p, t - o.r, p_avg, float(n))
            if isinstance(s, complex) or s != s or s in (math.inf, -math.inf):
                raise ValueError("priority is not a finite real number")
            if s > best_score:
                best, best_score = i, s
        queue.remove(best)
        t += orders[best].p
        late = orders[best].w * max(0.0, t - orders[best].d)
        cost += late
        if detail is not None:
            detail.append((orders[best], late))
    return cost


# Disjoint seed ranges: the search sees TRAIN, model selection uses VALIDATION, the final claim uses TEST (never touched by any method).
SPLITS = {"train": range(1000, 1024), "steer": range(1500, 1506), "validation": range(2000, 2024), "test": range(3000, 3060),
          "shift": range(4000, 4030)}   # "shift": a harder, out-of-distribution day (higher load, tighter due times)


def instances(split: str) -> list[tuple[Order, ...]]:
    if split == "shift":
        return [make_instance(s, load=1.05, tightness=1.8) for s in SPLITS[split]]
    return [make_instance(s) for s in SPLITS[split]]


# Classical dispatch rules from the scheduling literature, as expressions in the same language the search uses.
CLASSICAL = {
    "FIFO": "-(t - wait)",                                   # earliest arrival first (t - wait = r)
    "EDD": "-d",                                             # earliest due date
    "WSPT": "w / p",                                         # weighted shortest processing time
    "ATC": "w / p * exp(-max(0, slack) / (2.0 * p_avg))",    # apparent tardiness cost (Vepsalainen & Morton, 1987), look-ahead k = 2
}
