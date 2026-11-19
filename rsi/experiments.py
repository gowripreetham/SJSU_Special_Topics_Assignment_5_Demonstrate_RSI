"""Run every method under the same budget on fresh seeds, score the chosen rules on held-out data, write results/.

    python -m rsi.experiments                 # full study: 20 seeds × 2,400 shifts per run (about two minutes)
    python -m rsi.experiments --seeds 5       # quicker

Protocol
  · every run starts from the same rule (EDD) and may simulate exactly `budget` shifts on train/steer
  · the rule a run reports is its best by TRAIN score — the only thing the method saw
  · validation, test and the shifted-day set are computed afterwards and never fed back
  · seeds 100+ are used here; method development used seeds 0–9 and the validation split only
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from . import metrics, swarm, workflow
from .env import Env, held_out
from .methods.aide_tree import aide_tree
from .methods.baselines import naive_rsi, random_search
from .methods.budget_rsi import budget_rsi
from .methods.common import FULL, best_of
from .methods.so_rsi import so_rsi

OUT = Path(__file__).resolve().parents[1] / "results"
METHODS = {
    "Random search": random_search,
    "Naive RSI (hill climb)": naive_rsi,
    "AIDE-style tree search": aide_tree,
    "Budget RSI (readout)": budget_rsi,
    "Budget RSI — no edit features": lambda env, rng: budget_rsi(env, rng, readout="parent"),
    "SO-RSI (diagnostic)": so_rsi,
}
HEADLINE = ["AIDE-style tree search", "Budget RSI (readout)", "SO-RSI (diagnostic)"]
CHECKPOINTS = [FULL * k for k in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)]


def one_run(fn, seed: int, budget: int) -> dict:
    env = Env(budget=budget)
    nodes = fn(env, random.Random(seed))
    best = best_of(nodes)
    curve = []
    for cp in [c for c in CHECKPOINTS if c <= budget]:                       # incumbent (by train score) at each checkpoint, scored on test afterwards
        seen = [h for h in env.history if h[0] <= cp]
        curve.append(held_out(seen[-1][2])["test"] if seen else None)
    return {"seed": seed, "rule": best.rule, "train": best.score, **held_out(best.rule), "evaluated": len(nodes), "crashed": sum(not n.result.ok for n in nodes),
            "spent": env.spent, "curve": curve, "meta": nodes[0].meta.get("stats", {})}


def readout_quality(seeds: int = 10, per_seed: int = 60) -> dict:
    """Lab 10.04 — is the readout any good at ranking edits? Label EVERY proposed edit (no selection), fit on the first 60%,
    and measure on the rest: rank correlation between predicted and actual child cost, and how often the top prediction is a real improvement."""
    import numpy as np
    from . import program
    from .methods.budget_rsi import CLIP, features, fit_readout
    from .methods.common import START, evaluate_node
    rhos, hit_top, hit_any, crash_top, crash_any = [], [], [], [], []
    for s in range(seeds):
        rng, env, nodes = random.Random(500 + s), Env(budget=10**9), []
        evaluate_node(env, nodes, START, "start", None)
        rows = []
        while len(rows) < per_seed:
            parent = rng.choice([n for n in nodes if n.result.ok and n.score < 0.8])
            op = rng.choice(program.IMPROVE_OPS + ["draft"])
            rule = program.apply(op, parent.rule, rng)
            if any(rule == n.rule for n in nodes):
                continue
            child = evaluate_node(env, nodes, rule, op, parent.id)
            rows.append((features(parent, rule, op, "full"), child.score, child.score < parent.score, not child.result.ok))
        cut = int(per_seed * 0.6)
        w = fit_readout([r[0] for r in rows[:cut]], [r[1] for r in rows[:cut]])
        test = rows[cut:]
        pred, actual = np.array([float(np.dot(w, r[0])) for r in test]), np.minimum(np.array([r[1] for r in test]), CLIP)
        rank = lambda a: np.argsort(np.argsort(a)).astype(float)   # noqa: E731
        rhos.append(float(np.corrcoef(rank(pred), rank(actual))[0, 1]))
        top = np.argsort(pred)[:max(1, len(test) // 4)]                       # the quarter the readout likes best
        hit_top.append(statistics.mean(test[i][2] for i in top)); hit_any.append(statistics.mean(r[2] for r in test))
        crash_top.append(statistics.mean(test[i][3] for i in top)); crash_any.append(statistics.mean(r[3] for r in test))
    return {"spearman_mean": statistics.mean(rhos), "spearman_sd": statistics.stdev(rhos), "improve_rate_top_quarter": statistics.mean(hit_top), "improve_rate_all": statistics.mean(hit_any),
            "crash_rate_top_quarter": statistics.mean(crash_top), "crash_rate_all": statistics.mean(crash_any), "edits_per_seed": per_seed, "seeds": seeds}


def main(seeds: int = 20, budget: int = 2400) -> dict:
    t0 = time.time()
    ref = metrics.references()
    start, target = ref["EDD"]["test"], ref["ATC (k tuned on train)"]["test"]
    runs = {name: [one_run(fn, 100 + s, budget) for s in range(seeds)] for name, fn in METHODS.items()}
    table = {}
    for name, rs in runs.items():
        row = {k: metrics.mean_sd([r[k] for r in rs]) for k in ("train", "validation", "test", "shift")}
        row["rsi_index"] = metrics.mean_sd([metrics.rsi_index(r["test"], start, target) for r in rs])
        row["beats_reference"] = sum(r["test"] < target for r in rs)
        row["median_test"] = statistics.median(r["test"] for r in rs)
        row["test_crashes"] = sum(r["test"] > 5 for r in rs)
        row["gen_gap"] = statistics.mean(r["test"] - r["train"] for r in rs)
        row["evaluated"] = statistics.mean(r["evaluated"] for r in rs)
        row["crashed"] = statistics.mean(r["crashed"] for r in rs)
        table[name] = row
    naive = [r["test"] for r in runs["Naive RSI (hill climb)"]]
    paired = {name: metrics.paired_bootstrap([r["test"] for r in runs[name]], naive) for name in METHODS if name != "Naive RSI (hill climb)"}
    paired["Budget RSI: readout vs no-features"] = metrics.paired_bootstrap([r["test"] for r in runs["Budget RSI (readout)"]], [r["test"] for r in runs["Budget RSI — no edit features"]])

    # behaviour: how often does a first-order trace thrash, with and without the second-order layer?
    behaviour = {}
    for label, diag in (("naive (monitor only)", False), ("SO-RSI", True)):
        stats = [so_rsi(Env(budget=budget), random.Random(100 + s), diagnose=diag)[0].meta for s in range(seeds)]
        behaviour[label] = {k: statistics.mean(m["stats"].get(k, 0) / m["steps"] * 100 for m in stats) for k in ("recurrence", "opposing_edit", "expectation_mismatch")} | {
            "diagnoses_per_run": statistics.mean(m["stats"].get("diagnoses", 0) for m in stats)}

    # groups 03 and 04: coordination, adaptive workflow, self-tuning — same budget, same seeds
    extra = {"Swarm — 3 independent workers": [held_out(best_of(swarm.independent(budget, 100 + s)[0]).rule) for s in range(seeds)],
             "Swarm — 3 coordinated workers": [held_out(swarm.coordinated(budget, 100 + s)[0][0].rule) for s in range(seeds)],
             "Adaptive workflow (meta-agent)": [held_out(best_of(workflow.adaptive_flow(Env(budget=budget), random.Random(100 + s))).rule) for s in range(seeds)],
             "Tree search + Self-HPO": [held_out(workflow.self_hpo(budget, 100 + s)[0].rule) for s in range(seeds)]}
    extra_table = {k: {"test": metrics.mean_sd([r["test"] for r in v]), "rsi_index": statistics.mean(metrics.rsi_index(r["test"], start, target) for r in v)} for k, v in extra.items()}
    extra_paired = {"coordinated − independent": metrics.paired_bootstrap([r["test"] for r in extra["Swarm — 3 coordinated workers"]], [r["test"] for r in extra["Swarm — 3 independent workers"]]),
                    "self-HPO − default tree search": metrics.paired_bootstrap([r["test"] for r in extra["Tree search + Self-HPO"]], [r["test"] for r in runs["AIDE-style tree search"]]),
                    "adaptive workflow − naive": metrics.paired_bootstrap([r["test"] for r in extra["Adaptive workflow (meta-agent)"]], naive)}

    # lab 10.05 — my extension of Budget RSI, on a SEPARATE block of seeds (200+) so it is not judged on the seeds that motivated it
    ext = {label: [held_out(best_of(budget_rsi(Env(budget=budget), random.Random(200 + s), readout=ro)).rule)["test"] for s in range(seeds)]
           for label, ro in (("readout picks the top edit", "full"), ("readout only filters out the worst half", "filter"), ("no edit features (random sibling)", "parent"))}
    extension = {k: {"mean": statistics.mean(v), "median": statistics.median(v), "test_crashes": sum(x > 5 for x in v)} for k, v in ext.items()}
    # cost: what does more (or less) budget buy? 10 seeds per cell
    sweep = {b: {name: statistics.median(held_out(best_of(METHODS[name](Env(budget=b), random.Random(300 + s))).rule)["test"] for s in range(10))
                 for name in ("Random search", "Naive RSI (hill climb)", *HEADLINE)} for b in (600, 2400, 6000)}
    quality = readout_quality()

    half = CHECKPOINTS.index(FULL * 50)                                                             # 1,200 shifts — a post-hoc look, added after seeing the curves
    mid = {name: metrics.paired_bootstrap([r["curve"][half] for r in runs[name]], [r["curve"][half] for r in runs["Naive RSI (hill climb)"]]) for name in HEADLINE}

    best_overall = min((r for rs in runs.values() for r in rs), key=lambda r: r["validation"])      # chosen on VALIDATION, reported on test
    result = {"config": {"seeds": seeds, "first_seed": 100, "budget_shifts": budget, "full_evaluations": budget // FULL}, "references": ref, "table": table, "paired_vs_naive": paired,
              "behaviour": behaviour, "budget_extension": extension, "budget_sweep": sweep, "readout_quality": quality, "half_budget_vs_naive": mid, "extra": extra_table, "extra_paired": extra_paired, "best_rule_by_validation": best_overall, "runs": runs, "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    (OUT / "SUMMARY.md").write_text(summary(result))
    plots(result)
    return result


def summary(r: dict) -> str:
    f = lambda ms: f"{ms[0]:.3f} ± {ms[1]:.3f}"   # noqa: E731
    c = r["config"]
    L = [f"# Results — {c['seeds']} seeds × {c['budget_shifts']} simulated shifts per run ({c['full_evaluations']} full evaluations)", "",
         "Relative cost = total weighted tardiness ÷ FIFO's on the same shifts (lower is better). Mean ± sd over seeds. Rule reported per run = best by train score.", "",
         "## Hand-designed reference rules (held-out test)", "", "| Rule | test | shifted day |", "|---|---|---|"]
    L += [f"| {k} | {v['test']:.3f} | {v['shift']:.3f} |" for k, v in r["references"].items()]
    L += ["", "## Methods at equal budget", "", "| Method | train | validation | **test** (mean ± sd) | test median | shifted day | RSI-Index (median run) | runs beating tuned ATC | runs whose rule crashes on test | rules evaluated | crashed in search |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    s0, tgt = r["references"]["EDD"]["test"], r["references"]["ATC (k tuned on train)"]["test"]
    L += [f"| {k} | {f(v['train'])} | {f(v['validation'])} | **{f(v['test'])}** | {v['median_test']:.3f} | {f(v['shift'])} | {metrics.rsi_index(v['median_test'], s0, tgt):.2f} | {v['beats_reference']}/{c['seeds']} | {v['test_crashes']}/{c['seeds']} | {v['evaluated']:.0f} | {v['crashed']:.0f} |" for k, v in r["table"].items()]
    L += ["", "A rule that divides by something that is zero on an unseen shift crashes there and is scored 9.99 — that is what inflates a mean. Medians are robust to it."]
    L += ["", "## Paired differences in test cost (same seeds; negative = lower cost; 95% bootstrap interval)", "", "| Comparison | mean difference | 95% interval | reading |", "|---|---|---|---|"]
    read = lambda lo, hi: "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"   # noqa: E731
    L += [f"| {k} − naive RSI | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read(lo, hi)} |" if "vs" not in k else f"| {k} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read(lo, hi)} |" for k, (m, lo, hi) in r["paired_vs_naive"].items()]
    L += ["", "## Half-way through the budget (1,200 shifts) — exploratory, added after looking at the learning curves", "", "| Comparison | mean difference | 95% interval | reading |", "|---|---|---|---|"]
    L += [f"| {k} − naive RSI | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read(lo, hi)} |" for k, (m, lo, hi) in r["half_budget_vs_naive"].items()]
    L += ["", "## Behaviour: anomalies per 100 steps", "", "| Loop | recurrence | opposing edits | expectation mismatch | diagnoses per run |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['recurrence']:.1f} | {v['opposing_edit']:.1f} | {v['expectation_mismatch']:.1f} | {v['diagnoses_per_run']:.1f} |" for k, v in r["behaviour"].items()]
    q = r["readout_quality"]
    L += ["", "## Budget RSI — does the readout rank edits? (lab 10.04)", "",
          f"Every proposed edit labelled, {q['edits_per_seed']} edits × {q['seeds']} seeds, fit on the first 60%, measured on the rest.", "",
          f"- Spearman rank correlation, predicted vs actual child cost: **{q['spearman_mean']:.2f} ± {q['spearman_sd']:.2f}**",
          f"- Share of edits that improve on their parent: {q['improve_rate_top_quarter']:.0%} in the readout's top quarter vs {q['improve_rate_all']:.0%} overall",
          f"- Share of edits that crash: {q['crash_rate_top_quarter']:.0%} in the readout's top quarter vs {q['crash_rate_all']:.0%} overall",
          "", "## Budget RSI — extension (lab 10.05), separate seeds 200+", "", "| Variant | test mean | test median | runs whose rule crashes on test |", "|---|---|---|---|"]
    L += [f"| {k} | {v['mean']:.3f} | {v['median']:.3f} | {v['test_crashes']}/{c['seeds']} |" for k, v in r["budget_extension"].items()]
    L += ["", "## What budget buys — median test cost, 10 seeds (seeds 300+)", "", "| Shifts | " + " | ".join(next(iter(r["budget_sweep"].values()))) + " |", "|---|" + "---|" * len(next(iter(r["budget_sweep"].values())))]
    L += [f"| {b} | " + " | ".join(f"{v:.3f}" for v in row.values()) + " |" for b, row in r["budget_sweep"].items()]
    L += ["", "## Coordination, adaptive workflow, self-tuning (same budget and seeds)", "", "| Variant | test | RSI-Index |", "|---|---|---|"]
    L += [f"| {k} | {f(v['test'])} | {v['rsi_index']:.2f} |" for k, v in r["extra"].items()]
    L += ["", "| Comparison | mean difference | 95% interval | reading |", "|---|---|---|---|"]
    L += [f"| {k} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read(lo, hi)} |" for k, (m, lo, hi) in r["extra_paired"].items()]
    b = r["best_rule_by_validation"]
    L += ["", "## Best rule found (chosen by validation score across all runs)", "", f"`{b['rule']}`", "", f"validation {b['validation']:.3f} · test {b['test']:.3f} · shifted day {b['shift']:.3f}", ""]
    return "\n".join(L)


def plots(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
    ink, muted, grid = "#1d1c1a", "#52514e", "#e1e0d9"
    xs = [c for c in CHECKPOINTS if c <= r["config"]["budget_shifts"]]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ends = []
    for name, rs in r["runs"].items():
        ys = [statistics.median(run["curve"][i] for run in rs) for i in range(len(xs))]
        ax.plot(xs, ys, color=colors[name], lw=2 if name in HEADLINE else 1.4, ls="-" if name in HEADLINE or name.startswith("Naive") else (0, (4, 2)), label=name)
        ends.append((ys[-1], name))
    ref = r["references"]["ATC (k tuned on train)"]["test"]
    ax.axhline(ref, color=muted, lw=1, ls=":")
    ax.text(xs[-1], ref + 0.003, "tuned ATC (hand-designed reference)", color=muted, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("simulated shifts spent", color=muted); ax.set_ylabel("held-out test cost relative to FIFO (lower is better)", color=muted)
    ax.set_title("Median test cost of the incumbent rule as the budget is spent", loc="left", fontsize=11, color=ink)
    ax.grid(axis="y", color=grid, lw=0.7); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylim(ref - 0.012, None)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=140)
    names = list(r["runs"])
    for i, name in enumerate(names):
        vals = [run["test"] for run in r["runs"][name]]
        jitter = [(j % 5 - 2) * 0.05 for j in range(len(vals))]
        shown = [min(v, 0.72) for v in vals]                                   # a crashed rule (9.99) is drawn at the right edge
        m = statistics.median(vals)
        ax.scatter(shown, [i + d for d in jitter], s=16, color=colors[name], alpha=0.55, linewidths=0)
        ax.plot([m, m], [i - 0.3, i + 0.3], color=ink, lw=2)
        ax.text(0.735, i, f"median {m:.3f}" + (f"  ({sum(v > 5 for v in vals)} crashed →)" if any(v > 5 for v in vals) else ""), va="center", fontsize=8, color=ink)
    ax.axvline(ref, color=muted, lw=1, ls=":"); ax.text(ref, len(names) - 0.45, " tuned ATC", fontsize=8, color=muted, va="bottom")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8, color=ink); ax.invert_yaxis()
    ax.set_xlabel("held-out test cost relative to FIFO — one dot per seed, bar = median", color=muted); ax.set_xlim(0.42, 0.9)
    ax.set_title("Final rule of every run", loc="left", fontsize=11, color=ink)
    ax.grid(axis="x", color=grid, lw=0.7); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(colors=muted, labelsize=8, length=0)
    fig.tight_layout(); fig.savefig(OUT / "final_test_cost.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2400)
    a = ap.parse_args()
    res = main(a.seeds, a.budget)
    print(summary(res))
    print(f"[{res['seconds']}s] wrote results/results.json, SUMMARY.md, learning_curves.png, final_test_cost.png")
