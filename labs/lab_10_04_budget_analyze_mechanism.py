"""Lab 10.04 — Budget RSI, analyse the mechanism: is the readout actually informative?
    python -m labs.lab_10_04_budget_analyze_mechanism"""
import numpy as np

from rsi.experiments import readout_quality
from rsi.methods.budget_rsi import fit_readout

from ._util import header, run, table

NAMES = ["bias", "parent cost", "size growth", "variables introduced", "variables removed", "uses w and p", "due-date aware", "adds exp", "constants added", "near size limit"]
if __name__ == "__main__":
    header("10.04", "does the readout rank edits?", "If the readout were useless, what rank correlation and what improve-rate in its top quarter would you expect?")
    q = readout_quality(seeds=6)
    print(f"Out-of-sample, {q['edits_per_seed']} labelled edits × {q['seeds']} seeds (every proposed edit labelled, no selection):")
    print(f"  Spearman(predicted, actual)            {q['spearman_mean']:+.2f} ± {q['spearman_sd']:.2f}      (0 = useless, 1 = perfect)")
    print(f"  edits that improve on their parent     {q['improve_rate_top_quarter']:.0%} in the readout's top quarter  vs  {q['improve_rate_all']:.0%} of all edits")
    print(f"  edits that crash                       {q['crash_rate_top_quarter']:.0%} in the readout's top quarter  vs  {q['crash_rate_all']:.0%} of all edits\n")
    from rsi.methods.budget_rsi import budget_rsi, features
    _, nodes = run(budget_rsi, seed=0, readout="none", k=1)
    X = [features(nodes[n.parent], n.rule, n.op, "full") for n in nodes[1:]]
    w = fit_readout(X, [n.score for n in nodes[1:]])
    table([[name, f"{c:+.3f}"] for name, c in sorted(zip(NAMES, w), key=lambda t: -abs(t[1]))], ["feature", "weight (positive = predicts HIGHER cost)"])
    print("\nRead the two blocks together: a readout can rank edits better than chance and still be a poor PICKER if the best edits are rare outliers it cannot see.")
