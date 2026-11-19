"""Tests: the problem, the safe rule language, budget accounting, each method's mechanism, and the metrics. Small budgets; ~20 s."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi import metrics, problem, program, swarm, workflow  # noqa: E402
from rsi.env import CRASH, BudgetExhausted, Env, held_out  # noqa: E402
from rsi.methods import so_rsi as SO  # noqa: E402
from rsi.methods.aide_tree import aide_tree  # noqa: E402
from rsi.methods.baselines import naive_rsi, random_search  # noqa: E402
from rsi.methods.budget_rsi import budget_rsi, features, fit_readout, thompson_parent  # noqa: E402
from rsi.methods.common import FULL, START, GrammarProposer, LLMProposer, best_of  # noqa: E402

ALL = {"random": random_search, "naive": naive_rsi, "aide": aide_tree, "budget": budget_rsi, "so": SO.so_rsi, "flow": workflow.adaptive_flow}


# ---------------------------------------------------------------- problem
def test_instances_are_deterministic_and_splits_are_disjoint():
    assert problem.make_instance(7) == problem.make_instance(7) != problem.make_instance(8)
    seeds = [set(r) for r in problem.SPLITS.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(seeds) for b in seeds[i + 1:])


def test_simulation_is_a_valid_schedule_and_matches_a_hand_computed_case():
    o = (problem.Order(0, 10, 10, 1), problem.Order(0, 5, 5, 3), problem.Order(20, 2, 22, 1))
    assert problem.simulate(o, lambda p, w, d, t, slack, wait, p_avg, n: -d) == 5.0            # EDD: B(0-5) on time, A(5-15) late by 5 ×1, C(20-22) on time
    assert problem.simulate(o, lambda p, w, d, t, slack, wait, p_avg, n: -p) == 5.0
    assert problem.simulate(o, lambda p, w, d, t, slack, wait, p_avg, n: p) == 30.0            # A first: B finishes at 15, late by 10 × weight 3
    detail = []
    inst = problem.instances("train")[0]
    total = problem.simulate(inst, program.compile_rule("w / p"), detail)
    assert len(detail) == len(inst) and abs(sum(l for _, l in detail) - total) < 1e-6           # every order prepared exactly once
    with pytest.raises(ValueError):
        problem.simulate(o, lambda *a: float("nan"))


def test_textbook_ordering_of_classical_rules_holds_on_held_out_data():
    s = {k: held_out(v)["test"] for k, v in problem.CLASSICAL.items()}
    assert s["ATC"] < s["WSPT"] < s["EDD"] < s["FIFO"] == 1.0


# ---------------------------------------------------------------- rule language
@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "p.__class__", "[x for x in (1,)]", "open('f')", "w if p else d", "(lambda: 1)()", "p ** 99", "'a' * 3",
                                  "exec('1')", "w; p", "x + 1", "max(p, key=w)", "w +" , "1" + "+1" * 400])
def test_rule_whitelist_rejects(bad):
    with pytest.raises(program.RuleError):
        program.compile_rule(bad)
    assert Env(budget=10).evaluate(bad).error.startswith("rejected") and Env(budget=10).evaluate(bad).score == CRASH


def test_rules_run_without_builtins_and_operators_always_return_text():
    assert program.compile_rule("w / p * exp(-max(0, slack) / (2 * p_avg))")(10, 2, 30, 0, 20, 0, 10, 3.0) == pytest.approx(0.2 * 2.718281828 ** -1)
    assert program.compile_rule("exp(1000) + sqrt(-4) + log1p(-5)")(1, 1, 1, 1, 1, 1, 1, 1) > 0         # guarded functions never raise
    rng = random.Random(0)
    rule = START
    for i in range(300):
        op = rng.choice(list(program.OPERATORS) + ["debug"])
        child = program.apply(op, rule, rng)
        assert isinstance(child, str) and child
        try:
            program.compile_rule(child)
            rule = child if len(child) < 200 else START
        except program.RuleError:
            pass
    assert program.variables_used("w / p - slack") == {"w", "p", "slack"} and sorted(program.constants("2 * w + 0.5")) == [0.5, 2.0]
    assert "abs(" in program.apply("debug", "w / (p - p)", rng) and Env(budget=50).evaluate(program.apply("debug", "w / (p - p)", rng)).ok


def test_edit_features_need_no_execution():
    f = program.edit_features("-d", "-d * exp(-max(0, slack) / (2.0 * p_avg))", "wrap_urgency")
    assert len(f) == 8 and f[1] == 2.0 and f[5] == 1.0 and f[4] == 1.0
    assert program.edit_features("-d", "not a rule (", "x")[-1] == 1.0


# ---------------------------------------------------------------- environment
def test_budget_is_charged_per_shift_cached_and_enforced():
    env = Env(budget=FULL + 5)
    r = env.evaluate("w / p")
    assert env.spent == FULL and r.ok and len(r.per_instance) == FULL
    env.evaluate("w / p"); env.evaluate("w / p", limit=6)
    assert env.spent == FULL                                                        # cached: asking again is free
    assert env.evaluate("w / (p - p)").score == CRASH and env.spent == FULL + 1    # crashed on the first shift: one shift charged
    with pytest.raises(BudgetExhausted):
        env.evaluate("w / p + 1")
    assert env.spent == FULL + 5
    free = Env(budget=0)
    assert free.evaluate("w / p", "test").ok and free.spent == 0                   # held-out scoring is outside the budget


@pytest.mark.parametrize("name", list(ALL))
def test_every_method_respects_the_budget_and_never_gets_worse_than_its_start(name):
    env = Env(budget=480)
    nodes = ALL[name](env, random.Random(1))
    assert env.spent <= 480 and nodes[0].rule == START
    assert best_of(nodes).score <= nodes[0].score
    assert [h[1] for h in env.history] == sorted([h[1] for h in env.history], reverse=True)   # the recorded incumbent only ever improves
    again = ALL[name](Env(budget=480), random.Random(1))
    assert [n.rule for n in again] == [n.rule for n in nodes]                                  # same seed → same run


# ---------------------------------------------------------------- method mechanisms
def test_aide_tree_drafts_first_debugs_crashes_and_improves_the_best():
    class Crashy(GrammarProposer):
        def edit(self, op, parent, rng, hint=""):
            return "w / (p - p)" if op == "mul_term" else super().edit(op, parent, rng, hint)
    nodes = aide_tree(Env(budget=1200), random.Random(3), proposer=Crashy(), n_drafts=3)
    assert [n.op for n in nodes[:3]] == ["draft"] * 3
    debugs = [n for n in nodes if n.op == "debug"]
    assert debugs and all(not nodes[d.parent].result.ok for d in debugs) and any(d.result.ok for d in debugs)
    improves = [n for n in nodes if n.op in program.IMPROVE_OPS]
    assert improves and all(nodes[n.parent].result.ok for n in improves)


def test_budget_rsi_readout_is_fitted_and_used_only_after_enough_labels():
    nodes = budget_rsi(Env(budget=1200), random.Random(0), k=8, n_fit=12)
    children = nodes[1:]
    assert all(c.meta["predicted"] is None for c in children[:12]) and all(c.meta["predicted"] is not None for c in children[12:])
    assert max(c.meta["considered"] for c in children) > 1                               # several edits considered per evaluation
    w = fit_readout([[1, 0.6, 1], [1, 0.6, 0], [1, 0.5, 1], [1, 0.5, 0]], [0.5, 0.6, 0.4, 9.99])
    assert w.shape == (3,) and abs(float(w @ [1, 0.5, 0])) < 2                           # the crash label was clipped
    assert len(features(nodes[0], "w / p", "draft", "parent")) == 2 and len(features(nodes[0], "w / p", "draft", "full")) == 10
    picks = [thompson_parent(nodes, random.Random(i)).id for i in range(200)]
    assert len(set(picks)) > 1 and all(nodes[i].result.ok for i in picks)                # stochastic, and never a crashed node


def test_so_rsi_monitor_names_each_anomaly():
    step = lambda op, ok, revisit=False, expected=False, child=0.6: {"op": op, "parent": 0.5, "child": child, "accepted": ok, "revisit": revisit, "expected_gain": expected}  # noqa: E731
    assert SO.monitor([step("add_term", False)] * 2 + [step("swap_var", True)], {}) is None
    assert SO.monitor([step("tweak_const", False)] * 3, {}) == "recurrence:tweak_const"
    assert SO.monitor([step("add_term", True), step("drop_term", False, revisit=True)], {}) == "opposing_edit"
    assert SO.monitor([step("mul_term", False, expected=True, child=0.9)], {}) == "expectation_mismatch:mul_term"
    assert SO.monitor([step("mul_term", False, expected=True, child=0.51)], {}) is None


def test_so_rsi_probes_test_hypotheses_and_charge_the_budget():
    assert SO.probe_structure("-d") == ["w", "p", "wait"] and SO.probe_structure("w / p * exp(-slack) + wait") == []
    env = Env(budget=10_000)
    env.evaluate("-d")
    before = env.spent
    share = SO.probe_attribution(env, "-d")
    assert env.spent == before + SO.PROBE_SHIFTS and 0 <= share["heavy_share"] <= 1 and 0 <= share["long_share"] <= 1
    tests = SO.probe_counterfactuals(env, "w / p * exp(-max(0, slack) / (9.0 * p_avg))", [], random.Random(0))
    assert {t["hypothesis"] for t in tests} == {"const0_down", "const0_up"}
    down = next(t for t in tests if t["hypothesis"] == "const0_down")
    assert "4.5" in down["rule"] and down["gain"] > 0                                    # k = 9 is too large; halving it helps — the probe finds the direction
    nodes = SO.so_rsi(Env(budget=1500), random.Random(2))
    meta = nodes[0].meta
    assert meta["stats"].get("diagnoses", 0) == len(meta["memory"]) >= 1
    assert len({m["rule"] for m in meta["memory"]}) == len(meta["memory"])                # each incumbent is investigated once: the memory answers repeats
    assert any(n.meta.get("targeted") for n in nodes)
    assert SO.so_rsi(Env(budget=1500), random.Random(2), diagnose=False)[0].meta["memory"] == []


def test_llm_proposer_output_is_untrusted_text_that_must_pass_the_whitelist():
    rng = random.Random(0)
    assert LLMProposer(chat=lambda p: "Here you go:\n```python\nw / p * exp(-max(0, slack) / (2 * p_avg))\n```").edit("draft", START, rng) == "w / p * exp(-max(0, slack) / (2 * p_avg))"
    assert LLMProposer(chat=lambda p: "priority = w/p").edit("draft", START, rng) == "w / p"
    evil = LLMProposer(chat=lambda p: "__import__('os').system('rm -rf ~')").edit("draft", START, rng)
    assert Env(budget=5).evaluate(evil).error.startswith("rejected")                      # returned as text, refused by the environment, never executed
    program.compile_rule(LLMProposer(chat=lambda p: (_ for _ in ()).throw(TimeoutError())).edit("add_term", START, rng))   # network failure → offline operator
    env = Env(budget=240)
    nodes = naive_rsi(env, random.Random(0), proposer=LLMProposer(chat=lambda p: "w / p"))
    assert best_of(nodes).rule == "w / p" and env.stalled and env.spent == 2 * FULL          # a proposer that only repeats itself ends the run instead of hanging it


# ---------------------------------------------------------------- groups 03-05
def test_swarm_and_self_hpo_stay_inside_the_total_budget():
    best, spent = swarm.independent(720, seed=1)
    assert spent <= 720 and len(best) == 3
    best, spent, log = swarm.coordinated(720, seed=1)
    assert spent <= 720 and log["campaigns"] >= 1 and best[0].score <= 0.62
    node, spent, record = workflow.self_hpo(1200, seed=1)
    assert spent <= 1200 and record["winner"] in workflow.CONFIGS and len(record["rounds"]) == 2 and len(record["rounds"][1]["scores"]) == 3
    flow = workflow.adaptive_flow(Env(budget=1200), random.Random(1))
    assert flow[0].meta["routes"]["improve"] > 0 and flow[0].meta["routes"]["improve"] - sum(flow[0].meta["pulls"].values()) in (0, 1)   # the last step may run out of budget mid-evaluation


def test_metrics():
    assert metrics.rsi_index(0.609, 0.609, 0.457) == 0 and metrics.rsi_index(0.457, 0.609, 0.457) == pytest.approx(1) and metrics.rsi_index(0.40, 0.609, 0.457) > 1
    rule, score = metrics.tuned_atc()
    assert "exp" in rule and score <= Env(budget=10**6).evaluate(problem.CLASSICAL["ATC"]).score + 1e-9
    m, lo, hi = metrics.paired_bootstrap([0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6])
    assert m == pytest.approx(-0.1) and hi < 0
    m, lo, hi = metrics.paired_bootstrap([0.5, 0.7, 0.5, 0.7], [0.7, 0.5, 0.7, 0.5])
    assert lo < 0 < hi
