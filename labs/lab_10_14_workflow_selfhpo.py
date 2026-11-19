"""Lab 10.14 — Adaptive workflows, Self-HPO: the improver spends part of its budget tuning itself.
    python -m labs.lab_10_14_workflow_selfhpo"""
import json

from rsi import workflow
from rsi.methods.aide_tree import aide_tree

from ._util import BUDGET, DEV_SEEDS, best_of, header, held_out, mean, results, run

if __name__ == "__main__":
    header("10.14", "self-tuning the search", "30% of the budget goes to trial runs. When is that tax worth paying?")
    node, spent, record = workflow.self_hpo(BUDGET, seed=0)
    print(f"seed 0: {spent} shifts spent in total\n" + json.dumps(record, indent=1))
    tuned = mean(held_out(workflow.self_hpo(BUDGET, s)[0].rule)["validation"] for s in DEV_SEEDS)
    default = mean(held_out(best_of(run(aide_tree, s)[1]).rule)["validation"] for s in DEV_SEEDS)
    print(f"\nvalidation cost — self-tuned {tuned:.3f} vs default configuration {default:.3f} ({len(DEV_SEEDS)} development seeds)")
    r = results()
    if r:
        m, lo, hi = r["extra_paired"]["self-HPO − default tree search"]
        print(f"Reporting seeds, held-out test: self-HPO − default = {m:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]")
    print("Short trial runs are noisy rankings of configurations; the tax is only repaid if the configurations differ by more than that noise.")
