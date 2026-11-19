"""Lab 10.06 — SO-RSI, run it: watch the loop notice it is thrashing and stop to investigate.
    python -m labs.lab_10_06_so_run_demo"""
from rsi.methods.so_rsi import so_rsi

from ._util import best_of, header, held_out, run

if __name__ == "__main__":
    header("10.06", "SO-RSI demo", "Before running: which anomaly do you expect most often from a random hill climber — recurrence, opposing edits or expectation mismatch?")
    env, nodes = run(so_rsi, seed=3)
    meta = nodes[0].meta
    print(f"{meta['steps']} steps · anomalies {meta['stats']}\n")
    for m in meta["memory"][:4]:
        print(f"step {m['step']}: anomaly = {m['anomaly']}\n  incumbent: {m['rule']}\n  structure probe — the rule ignores: {m['missing'] or 'nothing'}")
        print(f"  attribution probe — share of weighted lateness from heavy orders {m['heavy_share']:.0%}, from long preparations {m['long_share']:.0%}")
        for t in sorted(m["tests"], key=lambda t: -t["gain"])[:4]:
            print(f"  hypothesis {t['hypothesis']:<18} gain on 6 shifts {t['gain']:+.4f}   {'→ queued' if t['gain'] > 0.005 else '  (not supported)'}   {t['rule'][:60]}")
        print()
    targeted = [n for n in nodes if n.meta.get("targeted")]
    print(f"targeted edits evaluated in full: {len(targeted)}, of which improved the incumbent: {sum(n.score < nodes[n.parent].score for n in targeted)}")
    b = best_of(nodes)
    print(f"best by train {b.score:.3f} → held-out {held_out(b.rule)}\n  {b.rule}")
