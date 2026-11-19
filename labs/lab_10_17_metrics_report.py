"""Lab 10.17 — Metrics, the full report: cost, generalisation, robustness and price — not one number.
    python -m labs.lab_10_17_metrics_report"""
from ._util import ROOT, header, results

if __name__ == "__main__":
    header("10.17", "read the whole table", "Before looking: which single column would mislead you most if it were the only one reported?")
    if not results():
        raise SystemExit("run `python -m rsi.experiments` first")
    print((ROOT / "results/SUMMARY.md").read_text())
