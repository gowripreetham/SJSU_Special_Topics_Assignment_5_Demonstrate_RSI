"""Lab 10.16 — Environments: the contract that makes methods comparable.
    python -m labs.lab_10_16_metrics_environment"""
from rsi import problem
from rsi.env import BudgetExhausted, Env

from ._util import header

if __name__ == "__main__":
    header("10.16", "the environment as a service", "List what a method could do to look better than it is. Which of those does the environment make impossible?")
    env = Env(budget=30)
    print("splits:", {k: len(v) for k, v in problem.SPLITS.items()}, "— only train and steer are chargeable; the rest are scored by the runner afterwards")
    r = env.evaluate("w / p")
    print(f"evaluate('w / p')            → cost {r.score:.3f}, spent {env.spent}")
    env.evaluate("w / p"); print(f"same rule again              → spent {env.spent} (cached)")
    print(f"evaluate('w / (p - p)')      → {env.evaluate('w / (p - p)').error!r}, spent {env.spent} (a crash costs what it used)")
    print(f"evaluate(\"__import__('os')\")  → {env.evaluate('__import__(chr(111)+chr(115))').error!r}, spent {env.spent} (never executed)")
    try:
        env.evaluate("w / p + wait")
    except BudgetExhausted:
        print(f"one more new rule            → BudgetExhausted at {env.spent}/{env.budget}")
    print("\nNot prevented: fitting the 24 train shifts. That is why the study reports validation, test and a shifted day.")
