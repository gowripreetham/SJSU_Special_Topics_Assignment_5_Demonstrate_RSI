"""Lab 10.05 — Budget RSI, extend it: use the readout to discard bad edits instead of crowning one.
    python -m labs.lab_10_05_budget_extend"""
from rsi.methods.budget_rsi import budget_rsi

from ._util import DEV_SEEDS, best_of, header, held_out, mean, results, run, table

if __name__ == "__main__":
    header("10.05", "extension: filter, don't pick", "Greedy use of a noisy predictor kills exploration. Does 'drop the predicted-worst half, choose randomly among the rest' do better?")
    rows = []
    for label, ro in (("readout picks the top edit", "full"), ("readout filters out the worst half", "filter"), ("no edit features (random sibling)", "parent")):
        best = [best_of(run(budget_rsi, s, readout=ro)[1]) for s in DEV_SEEDS]
        rows.append([label, f"{mean(b.score for b in best):.3f}", f"{mean(held_out(b.rule)['validation'] for b in best):.3f}"])
    table(rows, ["variant", "train", "validation"])
    print(f"\n({len(DEV_SEEDS)} development seeds.)")
    r = results()
    if r:
        print("\nOn the separate reporting seeds (200+), held-out test:")
        table([[k, f"{v['mean']:.3f}", f"{v['median']:.3f}", v["test_crashes"]] for k, v in r["budget_extension"].items()], ["variant", "mean", "median", "rules crashing on test"])
