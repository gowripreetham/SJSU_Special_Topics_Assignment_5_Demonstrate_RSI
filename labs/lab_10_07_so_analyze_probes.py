"""Lab 10.07 — SO-RSI, analyse the probes: do they find planted faults?
    python -m labs.lab_10_07_so_analyze_probes"""
import random

from rsi.env import Env
from rsi.methods import so_rsi as SO

from ._util import header

PLANTED = [("EDD: ignores weight and preparation time", "-d"), ("WSPT: ignores due dates", "w / p"),
           ("ATC with a look-ahead constant 6× too large", "w / p * exp(-max(0, slack) / (12.0 * p_avg))"), ("ATC, well tuned (control: nothing to find)", "w / p * exp(-max(0, slack) / (2.0 * p_avg))")]
if __name__ == "__main__":
    header("10.07", "probe a rule with a known fault", "For each planted fault, which hypothesis should come out on top? Predict before reading the output.")
    for label, rule in PLANTED:
        env = Env(budget=10**6)
        env.evaluate(rule)
        before = env.spent
        missing = SO.probe_structure(rule)
        tests = SO.probe_counterfactuals(env, rule, missing, random.Random(0))
        top = max(tests, key=lambda t: t["gain"]) if tests else None
        print(f"{label}\n  rule: {rule}\n  ignores: {missing or 'nothing'}   probes cost {env.spent - before} shifts")
        print(f"  best-supported hypothesis: {top['hypothesis']} (gain {top['gain']:+.4f})" + ("" if top["gain"] > 0.005 else "  → nothing supported") + "\n" if top else "  no hypotheses to test\n")
    print("Limit: a probe can only confirm a hypothesis someone thought to write. Faults outside the hypothesis list are invisible to it.")
