"""The environment: the only way a method can learn anything about a rule, with every evaluation charged to a budget.

Unit of cost = one simulated shift (one rule on one instance). A "full evaluation" is the whole train split (24 shifts).
The ledger is what makes the comparison between methods fair: every method gets the same number of shifts.

Scores are RELATIVE COST: total weighted tardiness divided by FIFO's on the same instances (FIFO = 1.00, lower is better).
A rule that fails to compile, crashes, or returns a non-finite value scores CRASH and still costs what it consumed.
Re-evaluating a rule is free (cached). To keep a proposer that only repeats itself from running forever, a run also ends
when STALL_LIMIT evaluations in a row cost nothing: the search has stalled.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import problem
from .program import RuleError, compile_rule

CRASH = 9.99   # the score of a rule that cannot be evaluated (worse than anything that runs)
STALL_LIMIT = 300


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Result:
    score: float                 # relative cost on the instances evaluated (lower is better); CRASH if it failed
    per_instance: list[float]    # relative cost per instance (empty on a crash)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class Env:
    budget: int                                         # shifts this run may simulate on train + steer
    spent: int = 0
    history: list[tuple[int, float, str]] = field(default_factory=list)   # (spent, train score, rule) each time the incumbent improves
    stalled: bool = False
    _idle: int = 0
    _cache: dict = field(default_factory=dict)
    _data: dict = field(default_factory=dict)

    def _split(self, name: str):
        if name not in self._data:
            inst = problem.instances(name)
            fifo = compile_rule(problem.CLASSICAL["FIFO"])
            self._data[name] = (inst, [max(problem.simulate(i, fifo), 1.0) for i in inst])
        return self._data[name]

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    def idle(self) -> None:
        """Called for every evaluation that cost nothing (and by methods when a step produced nothing new)."""
        self._idle += 1
        if self._idle >= STALL_LIMIT:
            self.stalled = True
            raise BudgetExhausted("stalled: no new rule in %d consecutive steps" % STALL_LIMIT)

    def evaluate(self, rule: str, split: str = "train", limit: int | None = None) -> Result:
        """Simulate `rule` on the first `limit` instances of a split (all if None). Charges the budget for train/steer;
        validation, test and shift are free here because methods never call them — only the experiment runner does."""
        inst, fifo = self._split(split)
        idx = range(len(inst) if limit is None else min(limit, len(inst)))
        charge = split in ("train", "steer")
        before = self.spent
        try:
            f = compile_rule(rule)
        except RuleError as e:
            if charge:
                self.idle()
            return Result(CRASH, [], f"rejected: {e}")                 # never ran: free
        costs = []
        for i in idx:
            key = (rule, split, i)
            if key not in self._cache:
                if charge:
                    if self.spent >= self.budget:
                        raise BudgetExhausted
                    self.spent += 1
                try:
                    self._cache[key] = problem.simulate(inst[i], f)
                except (ArithmeticError, ValueError, TypeError) as e:
                    self._cache[key] = f"{type(e).__name__}: {e}"
            if isinstance(self._cache[key], str):
                self._settle(charge, before)
                return Result(CRASH, [], self._cache[key])
            costs.append(self._cache[key])
        self._settle(charge, before)
        total_fifo = sum(fifo[i] for i in idx)
        return Result(sum(costs) / total_fifo, [c / fifo[i] for c, i in zip(costs, idx)])

    def _settle(self, charge: bool, before: int) -> None:
        if charge and self.spent == before:
            self.idle()
        elif charge:
            self._idle = 0

    def note_best(self, score: float, rule: str) -> None:
        if not self.history or score < self.history[-1][1]:
            self.history.append((self.spent, score, rule))


def held_out(rule: str) -> dict[str, float]:
    """Scores no method ever sees: validation (to choose between runs), test and the shifted day (to report)."""
    env = Env(budget=0)
    return {s: env.evaluate(rule, s).score for s in ("validation", "test", "shift")}
