"""Lab 10.01 — Read a frontier claim: separate what a source claims from what this repo shows.
    python -m labs.lab_10_01_claim_vs_evidence"""
from ._util import header, results, table

LEDGER = [
    ["AIDE-style tree search", "Searching a tree of complete solutions with draft / debug / improve beats linear iteration on ML-engineering tasks",
     "tree of rules, 3 operators, greedy parent choice, full journal", "LLM-written ML scripts → one-line dispatch rules; grammar proposer by default"],
    ["Budget RSI (arXiv 2610.29121)", "A linear readout over parent solve-rate + diff features picks which edit to evaluate and raises held-out score on coding benchmarks",
     "Thompson-sampled parent, K edits scored by a linear readout, only the top one evaluated", "8 features of one-line rules; ~100 labels per run; my own feature set"],
    ["SO-RSI (arXiv 2610.05701)", "Diagnosing why failures occur (anomaly detection → probes → inquiry memory) beats patch-and-retry on proof/verification tasks",
     "monitor for recurrence / opposing edits / expectation mismatch; probes; memory; targeted edits", "probes are hypothesis tests on 6 simulated shifts; edits come from a fixed operator set"],
]
if __name__ == "__main__":
    header("10.01", "claim vs evidence", "For each method: what does the source claim, what did I keep, what did I change — and what did I observe?")
    print("Source claims are quoted from the COURSE NOTES' summaries. I did not read or verify the papers; the two 2026 arXiv ids are as given there.\n")
    r = results()
    for name, claim, kept, changed in LEDGER:
        print(f"{name}\n  source claim (unverified here): {claim}\n  mechanism kept: {kept}\n  changed for the laptop: {changed}")
        if r:
            key = next(k for k in r["table"] if k.startswith(name.split(" (")[0][:9]))
            t = r["table"][key]
            m, lo, hi = r["paired_vs_naive"][key]
            print(f"  observed here: test cost {t['test'][0]:.3f} ± {t['test'][1]:.3f} (median {t['median_test']:.3f}); vs naive hill climbing {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print()
    print("An observation on a 40-order scheduling toy says nothing about the sources' benchmark results, in either direction." + ("" if r else "\n(run `python -m rsi.experiments` to fill in the observed column)"))
