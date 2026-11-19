"""Lab 10.13 — Adaptive workflows, the meta-agent: what did it learn about its own operators?
    python -m labs.lab_10_13_workflow_rsiflow"""
from rsi import workflow

from ._util import header, run, table

if __name__ == "__main__":
    header("10.13", "inside the meta-agent", "Which operator do you expect to earn the highest reward per evaluation — and will the bandit's pull counts follow it?")
    for budget in (600, 2400):
        _, nodes = run(workflow.adaptive_flow, seed=1, budget=budget)
        m = nodes[0].meta
        print(f"after {budget} shifts — routes {m['routes']}")
        table(sorted([[op, m["pulls"][op], f"{m['mean_reward'][op]:.4f}"] for op in m["pulls"]], key=lambda r: -r[1]), ["operator", "times chosen", "mean relative improvement"])
        print()
    print("Reward is sparse (most edits do not improve), so the estimates are noisy and the bandit keeps exploring — an honest limit of learning from ~100 steps.")
