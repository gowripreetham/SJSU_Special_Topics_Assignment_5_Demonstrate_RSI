"""Lab 10.11 — Scaling RSI, swarm size: split a FIXED budget across more workers.
    python -m labs.lab_10_11_scaling_swarm"""
from rsi import swarm

from ._util import BUDGET, DEV_SEEDS, header, held_out, mean, results, table

if __name__ == "__main__":
    header("10.11", "more agents, same budget", "Each worker gets budget ÷ N. At what N do you expect the per-worker budget to become the bottleneck?")
    rows = []
    for n in (1, 3, 6):
        ind = [held_out(min(swarm.independent(BUDGET, s, workers=n)[0], key=lambda x: x.score).rule)["validation"] for s in DEV_SEEDS]
        co = [held_out(swarm.coordinated(BUDGET, s, workers=n)[0][0].rule)["validation"] for s in DEV_SEEDS]
        rows.append([n, BUDGET // n, f"{mean(ind):.3f}", f"{mean(co):.3f}"])
    table(rows, ["workers", "shifts per worker", "independent (validation)", "coordinated (validation)"])
    r = results()
    if r:
        m, lo, hi = r["extra_paired"]["coordinated − independent"]
        print(f"\nReporting seeds, 3 workers, held-out test: coordinated − independent = {m:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]")
