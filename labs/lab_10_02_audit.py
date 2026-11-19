"""Lab 10.02 — Audit a result: run the checks a sceptical reader would run, on this repo's own study.
    python -m labs.lab_10_02_audit"""
from ._util import header, results, table

if __name__ == "__main__":
    header("10.02", "audit the study", "Which ways of fooling myself does the protocol rule out, and which remain?")
    r = results()
    if not r:
        raise SystemExit("run `python -m rsi.experiments` first")
    c, runs = r["config"], r["runs"]
    checks = [
        ("every run stayed within the budget", all(x["spent"] <= c["budget_shifts"] for rs in runs.values() for x in rs)),
        ("same seeds for every method (paired comparison)", len({tuple(x["seed"] for x in rs) for rs in runs.values()}) == 1),
        ("reported seeds differ from development seeds (0–9)", min(x["seed"] for rs in runs.values() for x in rs) >= 100),
        ("at least 20 seeds per method", c["seeds"] >= 20),
        ("the reported rule was chosen by TRAIN score, test computed afterwards", True),
        ("search could not have read test data: Env charges only train/steer and methods never request other splits", True),
        ("failures are in the table (crashes during search, rules crashing on test)", all("test_crashes" in v for v in r["table"].values())),
    ]
    table([[("PASS" if ok else "FAIL"), what] for what, ok in checks], ["", "check"])
    gap = {k: v["gen_gap"] for k, v in r["table"].items()}
    print("\nRemaining weaknesses (not fixed by the protocol):")
    print(f"  · train→test gap of the chosen rules: {min(gap.values()):+.3f} to {max(gap.values()):+.3f} — every method fits the 24 train shifts somewhat")
    print("  · one synthetic problem family; conclusions do not transfer to other tasks or to LLM proposers")
    print("  · method hyper-parameters were set on development seeds using the validation split — a small, real degree of freedom")
    print("  · two of the checks above are properties of the code, asserted here and verified in tests/test_rsi.py, not measured from the results file")
