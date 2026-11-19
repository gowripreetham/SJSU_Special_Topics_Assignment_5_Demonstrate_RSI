"""Lab 10.12 — Adaptive workflows, compare: a fixed loop vs a loop whose next step depends on its own feedback.
    python -m labs.lab_10_12_workflow_compare"""
from rsi import workflow
from rsi.methods.baselines import naive_rsi

from ._util import DEV_SEEDS, best_of, header, held_out, mean, run, table

if __name__ == "__main__":
    header("10.12", "fixed vs adaptive workflow", "Both loops have the same edit operators. What extra information does the adaptive one use?")
    fixed = [best_of(run(naive_rsi, s)[1]) for s in DEV_SEEDS]
    flows = [run(workflow.adaptive_flow, s)[1] for s in DEV_SEEDS]
    table([["fixed: pick an operator uniformly at random", f"{mean(held_out(b.rule)['validation'] for b in fixed):.3f}", "—"],
           ["adaptive: bandit + bottleneck routing", f"{mean(held_out(best_of(n).rule)['validation'] for n in flows):.3f}", str(flows[0][0].meta["routes"])]],
          ["workflow", "validation cost", "routes taken (seed 0)"])
