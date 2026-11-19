"""Lab 10.08 — SO-RSI, apply it: first-order vs second-order on the same seeds, by behaviour and by score.
    python -m labs.lab_10_08_so_apply"""
from rsi.methods.so_rsi import so_rsi

from ._util import DEV_SEEDS, best_of, header, held_out, mean, results, run, table

if __name__ == "__main__":
    header("10.08", "does diagnosis reduce thrashing, and does that buy a better rule?", "These are two different claims. Which one do you expect to hold more clearly?")
    rows = []
    for label, diag in (("first-order (monitor only)", False), ("SO-RSI", True)):
        out = [run(so_rsi, s, diagnose=diag)[1] for s in DEV_SEEDS]
        rate = lambda k: mean(n[0].meta["stats"].get(k, 0) / n[0].meta["steps"] * 100 for n in out)   # noqa: E731
        rows.append([label, f"{rate('recurrence'):.1f}", f"{rate('opposing_edit'):.1f}", f"{rate('expectation_mismatch'):.1f}", f"{mean(held_out(best_of(n).rule)['validation'] for n in out):.3f}"])
    table(rows, ["loop", "recurrence /100 steps", "opposing /100", "mismatch /100", "validation cost"])
    r = results()
    if r:
        m, lo, hi = r["paired_vs_naive"]["SO-RSI (diagnostic)"]
        print(f"\nReporting seeds (100+), held-out test: SO-RSI − naive = {m:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]")
        print("Behaviour on the reporting seeds:", {k: {a: round(b, 1) for a, b in v.items()} for k, v in r["behaviour"].items()})
