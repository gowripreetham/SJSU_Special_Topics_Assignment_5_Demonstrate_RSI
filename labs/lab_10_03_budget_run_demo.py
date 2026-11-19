"""Lab 10.03 — Budget RSI, run it: watch the readout choose which edit gets the expensive evaluation.
    python -m labs.lab_10_03_budget_run_demo"""
from rsi.methods.budget_rsi import budget_rsi

from ._util import best_of, header, held_out, run, table

if __name__ == "__main__":
    header("10.03", "Budget RSI demo", "Before running: will predicted cost and actual cost agree? Write down a guess for the rank correlation.")
    env, nodes = run(budget_rsi, seed=0, budget=1200)
    rows = [[n.id, n.parent, n.op, n.meta["considered"], "—" if n.meta["predicted"] is None else f"{n.meta['predicted']:.3f}", f"{n.score:.3f}" if n.result.ok else "CRASH", n.rule[:58]] for n in nodes[1:]]
    table(rows[:10] + [["…"] * 7] + rows[-12:], ["node", "parent", "edit", "edits considered", "predicted", "actual", "rule"])
    b = best_of(nodes)
    print(f"\n{len(nodes) - 1} full evaluations chose among {sum(n.meta['considered'] for n in nodes[1:])} proposed edits · {env.spent} shifts spent")
    print(f"best by train: {b.score:.3f}  →  held-out {held_out(b.rule)}\n  {b.rule}")
    print("\nThe first 12 rows have no prediction: those evaluations are the labels the readout is fitted on.")
