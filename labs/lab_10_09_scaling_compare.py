"""Lab 10.09 — Scaling RSI, compare designs: independent workers vs a coordinated team, on paper first.
    python -m labs.lab_10_09_scaling_compare"""
from rsi import swarm

from ._util import BUDGET, header, held_out, table

if __name__ == "__main__":
    header("10.09", "two ways to use three workers", "Same total budget. What can a coordinated team do that three independent runs cannot — and what does it risk?")
    table([["independent", "none", "private", "best of three", "diversity kept, effort duplicated"],
           ["coordinated", "leader assigns an edit family per worker", "best rule published; crashes shared", "leader's review each campaign", "less duplication; can converge too early"]],
          ["design", "orientation", "memory", "selection", "trade-off"])
    ind, spent_i = swarm.independent(BUDGET, seed=0)
    co, spent_c, log = swarm.coordinated(BUDGET, seed=0)
    print(f"\none seed — independent: workers' train costs {[round(n.score, 3) for n in ind]} ({spent_i} shifts) → validation {held_out(min(ind, key=lambda n: n.score).rule)['validation']:.3f}")
    print(f"           coordinated: train {co[0].score:.3f} ({spent_c} shifts), {log} → validation {held_out(co[0].rule)['validation']:.3f}")
