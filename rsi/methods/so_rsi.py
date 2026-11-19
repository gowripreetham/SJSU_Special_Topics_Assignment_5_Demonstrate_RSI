"""Method 3 — SO-RSI: second-order diagnostic inquiry.

The idea (as summarised in the course's notes on "Second-Order Problem Solving for Recursive Self-Improvement",
arXiv 2610.05701 — I worked from that summary, not from the paper): a first-order improver reacts to a bad score by
trying another patch. When that stops working it THRASHES — repeats the same kind of edit, or undoes its own changes.
A second-order improver watches its own trace for those patterns and, when one appears, stops patching and asks WHY:
    monitor   detect anomalies in the trace:  recurrence · opposing edits · expectation mismatch
    probe     run small, targeted diagnostics on the incumbent rule (their cost is charged to the same budget)
    remember  keep the findings in an inquiry memory that persists for the rest of the run
    act       choose edits that address the diagnosed mechanism, and stop making edits the evidence says are futile

Probes used here
    structure     which decision variables does the rule ignore?                      (reads the AST — free)
    attribution   where does the lateness come from: STAT/heavy orders, long preparations?   (re-simulates 6 shifts with detail)
    counterfactual  one lightweight test per hypothesis, 6 shifts each: "the rule is blind to w / p / slack / wait"
                    (does introducing it help?) and "this constant is too high / too low" (×0.5 and ×2)

diagnose=False gives the first-order baseline with the SAME monitor counting anomalies but never acting on them, so the
two can be compared on behaviour (how often they thrash) as well as on score.
"""
from __future__ import annotations
import ast
import random
from collections import Counter, deque

from .. import problem, program
from ..env import BudgetExhausted, Env
from .common import START, Node, default_proposer, evaluate_node

PROBE_SHIFTS = 6


def monitor(trace: list[dict], seen: dict[str, float]) -> str | None:
    """Look at the recent trace and name the anomaly, if any."""
    recent = trace[-6:]
    if not recent:
        return None
    last = recent[-1]
    fails = Counter(s["op"] for s in recent if not s["accepted"])
    if fails and fails.most_common(1)[0][1] >= 3 and last["op"] == fails.most_common(1)[0][0] and not last["accepted"]:
        return f"recurrence:{last['op']}"                                   # the same kind of edit keeps failing
    if last["revisit"]:
        return "opposing_edit"                                              # the edit produced a rule we have already evaluated: it undid earlier work
    if last["expected_gain"] and last["child"] > last["parent"] * 1.10:
        return f"expectation_mismatch:{last['op']}"                         # an edit that had worked before made things clearly worse
    return None


def probe_structure(rule: str) -> list[str]:
    used = program.variables_used(rule)
    return [v for v, covered in (("w", "w" in used), ("p", "p" in used), ("slack", bool(used & {"slack", "d"})), ("wait", bool(used & {"wait", "t"}))) if not covered]


def probe_attribution(env: Env, rule: str) -> dict[str, float]:
    """Share of weighted tardiness caused by heavy (STAT / chemo) orders and by long preparations, on a few train shifts."""
    f = program.compile_rule(rule)
    heavy = long_ = total = 0.0
    for inst in problem.instances("train")[:PROBE_SHIFTS]:
        if env.spent >= env.budget:
            raise BudgetExhausted
        env.spent += 1                                                       # a probe is a simulation: it is charged
        detail: list = []
        problem.simulate(inst, f, detail)
        for order, late in detail:
            total += late
            heavy += late * (order.w >= 5.0)
            long_ += late * (order.p >= 20.0)
    return {"heavy_share": heavy / total if total else 0.0, "long_share": long_ / total if total else 0.0}


def probe_counterfactuals(env: Env, rule: str, missing: list[str], rng: random.Random) -> list[dict]:
    """Lightweight tests of specific hypotheses, on PROBE_SHIFTS shifts each:
         "the rule is blind to <variable>"        → does introducing it lower cost?
         "constant #i is set too high / too low"  → is cost lower at ×0.5 or at ×2?
    Returns one record per hypothesis with the candidate rule and its measured gain (positive = better)."""
    base = env.evaluate(rule, "train", PROBE_SHIFTS).score                   # cached: the incumbent was already evaluated
    tests = [(f"introduce_{v}#{i}", _norm(cand)) for v in missing for i, cand in enumerate(program.introductions(rule, rng, v))]
    n_consts = len([c for c in program.constants(rule) if c not in (0, 1)])
    for idx in range(min(n_consts, 2)):
        tests += [(f"const{idx}_down", scale_constant(rule, idx, 0.5)), (f"const{idx}_up", scale_constant(rule, idx, 2.0))]
    return [{"hypothesis": name, "rule": cand, "gain": round(base - env.evaluate(cand, "train", PROBE_SHIFTS).score, 4)} for name, cand in tests]


def scale_constant(rule: str, idx: int, factor: float) -> str:
    tree = program.parse(rule)
    consts = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value not in (0, 1)]
    consts[idx].value = round(float(consts[idx].value) * factor, 4)
    return ast.unparse(tree)


def so_rsi(env: Env, rng: random.Random, proposer=None, diagnose: bool = True, ban_for: int = 8) -> list[Node]:
    proposer, nodes = proposer or default_proposer(), []
    trace: list[dict] = []
    seen: dict[str, float] = {}
    memory: list[dict] = []                    # inquiry memory: every diagnosis with its evidence
    queue: deque = deque()                     # targeted edits waiting to be tried: (op label, rule)
    banned: dict[str, int] = {}                # op -> step until which it is not proposed
    worked: Counter = Counter()
    stats = Counter()
    diagnosed: set[str] = set()                # incumbents already investigated
    try:
        incumbent = evaluate_node(env, nodes, START, "start", None)
        seen[incumbent.rule] = incumbent.score
        step = 0
        while True:
            step += 1
            if queue:
                op, rule = queue.popleft()
            else:
                ops = [o for o in program.IMPROVE_OPS if banned.get(o, 0) < step] or program.IMPROVE_OPS
                op = rng.choice(ops)
                rule = proposer.edit(op, incumbent.rule, rng)
            revisit = rule in seen and rule != incumbent.rule
            child = evaluate_node(env, nodes, rule, op, incumbent.id, targeted=op.startswith("targeted"))
            accepted = child.score < incumbent.score
            trace.append({"op": op, "parent": incumbent.score, "child": child.score, "accepted": accepted, "revisit": revisit, "expected_gain": worked[op] > 0})
            seen[rule] = child.score
            if accepted:
                worked[op] += 1
                incumbent = child
                queue.clear()                  # the diagnosis was about the old incumbent
            anomaly = monitor(trace, seen)
            if not anomaly:
                continue
            stats[anomaly.split(":")[0]] += 1
            if not diagnose or queue:
                continue
            # ---- second-order step: investigate before patching again (once per incumbent — the memory answers repeat questions)
            if incumbent.rule in diagnosed:
                if anomaly.startswith("recurrence"):
                    banned[anomaly.split(":")[1]] = step + ban_for          # remembered: this kind of edit is not working here
                continue
            diagnosed.add(incumbent.rule)
            finding = {"step": step, "anomaly": anomaly, "rule": incumbent.rule, "missing": probe_structure(incumbent.rule)}
            finding |= probe_attribution(env, incumbent.rule)
            finding["tests"] = probe_counterfactuals(env, incumbent.rule, finding["missing"], rng)
            memory.append(finding)
            stats["diagnoses"] += 1
            supported = sorted((t for t in finding["tests"] if t["gain"] > 0.005 and t["rule"] not in seen), key=lambda t: -t["gain"])
            for t in supported:                                             # act on mechanisms the evidence supports, strongest first
                queue.append((f"targeted:{t['hypothesis']}", t["rule"]))
            if not any(t["gain"] > 0.005 for t in finding["tests"] if t["hypothesis"].startswith("const")):
                banned["tweak_const"] = step + ban_for                      # evidence: the constants are not the problem — stop tuning them
            if anomaly.startswith("recurrence"):
                banned[anomaly.split(":")[1]] = step + ban_for
    except BudgetExhausted:
        pass
    if nodes:
        nodes[0].meta |= {"stats": dict(stats), "memory": memory, "steps": len(trace)}
    return nodes


def _norm(rule: str) -> str:
    try:
        return program.normalise(rule)
    except program.RuleError:
        return rule
