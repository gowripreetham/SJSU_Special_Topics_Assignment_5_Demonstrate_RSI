"""Lab 10.15 — Metrics, RSI-Index: how much of the gap to a strong reference did each method close?
    python -m labs.lab_10_15_metrics_rsi_index"""
from rsi import metrics

from ._util import header, results, table

if __name__ == "__main__":
    header("10.15", "RSI-Index", "Index = (start − achieved) ÷ (start − reference). What does an index of 0.7 mean in words? What would 1.2 mean?")
    ref = metrics.references()
    start, target = ref["EDD"]["test"], ref["ATC (k tuned on train)"]["test"]
    print(f"start = EDD, test cost {start:.3f} · reference = tuned ATC, test cost {target:.3f}\n")
    rows = [[k, f"{v['test']:.3f}", f"{metrics.rsi_index(v['test'], start, target):.2f}"] for k, v in ref.items()]
    r = results()
    if r:
        rows += [[k, f"{v['median_test']:.3f} (median run)", f"{metrics.rsi_index(v['median_test'], start, target):.2f}"] for k, v in r["table"].items()]
        rows += [[k, f"{v['test'][0]:.3f} (mean)", f"{v['rsi_index']:.2f}"] for k, v in r["extra"].items()]
    table(rows, ["rule / method", "test cost", "RSI-Index"])
    print("\nThe index depends entirely on the chosen reference. Against a weaker reference (WSPT) every method here would score above 1.")
