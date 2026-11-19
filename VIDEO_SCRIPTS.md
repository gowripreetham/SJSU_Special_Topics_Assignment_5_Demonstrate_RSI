# Video scripts — Assignment 5 (three videos, one per method, ≈7 min each)

**[SCREEN]** = what to show; plain text = what to say. Run `python -m rsi.experiments` once beforehand so `results/` exists.
Video 1 carries the shared introduction (problem, environment, protocol); videos 2 and 3 start with a 20-second recap.
If you have an OpenRouter key, add one segment per video running a lab with `RSI_LLM=1` to show an LLM writing the edits.

## Video 1 — Setup, baselines and AIDE-style tree search (labs 10.01–10.02, 10.15–10.17)

- **0:00** [README top] "Assignment 5: three recursive self-improvement methods on one problem I chose — sequencing a hospital pharmacy's compounding queue. The thing that improves is the dispatch rule: a one-line program."
- **0:30** [`rsi/problem.py`: docstring, `simulate`, `SPLITS`, `CLASSICAL`] "Orders have a prep time, due time and clinical weight. Cost is weighted tardiness. Five disjoint sets of shifts: train for the search, validation for my design choices, test and a shifted day for the claims. Four textbook rules as references."
- **1:20** [`rsi/program.py`: `parse`, `ALLOWED`, an operator, `edit_features`] "A rule is parsed and every AST node checked against a whitelist — no imports, no attributes, no loops. So a rule written by an LLM can't do anything but arithmetic. Edits are small source-to-source operators."
- **2:10** [RUN `python -m labs.lab_10_16_metrics_environment`] "The environment charges one unit per simulated shift. Repeats are free, crashes cost what they used, and the budget is hard."
- **2:50** [`rsi/methods/baselines.py`] "Two baselines: random search, and first-order hill climbing — naive RSI."
- **3:20** [`rsi/methods/aide_tree.py`] "Method one, AIDE-style tree search. Every node is a complete rule with a score. Three operators: draft a new root, debug a crashed leaf — to a bounded depth — and improve the best node, with a little exploration. The whole journal is kept."
- **4:20** [RUN `python -m labs.demo_aide_tree`; then `python -m pytest tests -q -k aide`] narrate drafts → improve → a crash → debug.
- **5:10** [`results/SUMMARY.md` main table + `final_test_cost.png`] "Twenty seeds, same budget. Starting rule 0.609; hill climbing 0.512; tree search 0.523 — the interval for the difference includes zero. And look at the dotted line: tuned ATC at 0.457. No run beat it."
- **6:00** [RUN `python -m labs.lab_10_02_audit`, `python -m labs.lab_10_15_metrics_rsi_index`] "An audit of my own study, and the RSI-Index: tree search closed about 56% of the gap from the start to the textbook reference."
- **6:40** "Takeaway: the tree's diversity didn't pay at this budget on this problem. In the separate budget sweep it looked better at small budgets — but that's ten seeds, so I don't lean on it."

## Video 2 — Budget RSI: scoring edits before running them (labs 10.03–10.05)

- **0:00** Recap in two sentences. "Method two asks: evaluation is the expensive step, so can we choose which edit to evaluate without running it?"
- **0:30** [`rsi/methods/budget_rsi.py` docstring, `thompson_parent`, `features`, `fit_readout`, main loop] "Pick a parent by Thompson sampling. Propose eight edits — free. Score each with a linear readout over the parent's cost and eight edit features. Evaluate only the top one. Its true cost becomes a new label. I built this from the course's summary of the paper, not the paper itself."
- **1:50** [RUN `python -m labs.lab_10_03_budget_run_demo`] "The first twelve evaluations have no prediction — they're the training labels. After that, compare the predicted and actual columns." Point out the miscalibration.
- **3:00** [RUN `python -m labs.lab_10_04_budget_analyze_mechanism`] "Is the readout informative? Out of sample the rank correlation is about 0.36 — better than chance. But only 20% of its favourite quarter are real improvements, against 18% overall. And the weights: 'due-date aware' strongly predicts lower cost, which is sensible."
- **4:10** [`results/SUMMARY.md` Budget RSI rows] "And here's the result I didn't expect: picking the readout's top edit made the search worse than picking a sibling at random — 0.532 versus 0.487 on train. A decent ranker, a bad picker: improvements are rare outliers, and always taking the safe prediction kills exploration."
- **5:00** [`final_test_cost.png`, the crashed dot] "One run chose a rule that divides by something that's never zero on train but is zero on a test shift. It crashes on unseen data. That's why the mean is 1.015 and why I report medians too."
- **5:40** [RUN `python -m labs.lab_10_05_budget_extend`] "My extension: use the readout as a filter — drop the predicted-worst half, choose randomly among the rest. On separate seeds that recovers to the no-features level, not above."
- **6:30** "Takeaway: the mechanism works as described, but in my adaptation it doesn't buy anything, and I can say why. With an LLM proposer the costs change — proposing isn't free any more."

## Video 3 — SO-RSI: diagnose before patching; plus coordination, workflows and Self-HPO (labs 10.06–10.14)

- **0:00** Recap. "Method three: a first-order improver that isn't making progress thrashes. A second-order improver notices and asks why."
- **0:30** [`rsi/methods/so_rsi.py`: docstring, `monitor`] "The monitor watches the trace for three anomalies: recurrence, opposing edits, expectation mismatch."
- **1:10** [`probe_structure`, `probe_attribution`, `probe_counterfactuals`] "Then probes. Structure: which variables does the rule ignore? Attribution: where does the lateness come from? Counterfactuals: one cheap test per hypothesis on six shifts. Probes are charged to the same budget."
- **2:00** [main loop: `diagnosed`, `memory`, `queue`, `banned`] "Findings go to an inquiry memory; each incumbent is investigated once. Supported hypotheses become targeted edits; unsupported ones ban that kind of edit for a while."
- **2:40** [RUN `python -m labs.lab_10_06_so_run_demo`] read one diagnosis aloud.
- **3:30** [RUN `python -m labs.lab_10_07_so_analyze_probes`] "Planted faults: EDD ignores weight — the probe's top hypothesis is 'introduce w'. ATC with a constant six times too large — 'constant down'. Well-tuned ATC — nothing supported."
- **4:20** [RUN `python -m labs.lab_10_08_so_apply`; `learning_curves.png`; SUMMARY 'Half-way' and 'Behaviour' tables] "Two separate claims. Behaviour: recurrence drops from 15 to 12 per hundred steps. Score: at the full budget, minus 0.010 with an interval that includes zero. Half-way through the budget SO-RSI is ahead by 0.034 — but I looked at that point after seeing the curves, so it's exploratory. It's also the most consistent across seeds."
- **5:20** [`rsi/swarm.py`; RUN `python -m labs.lab_10_11_scaling_swarm`] "Scaling: three independent workers versus a coordinated team with a leader, shared findings and shared failures. Same total budget. Coordinated is nominally better, not clearly."
- **6:00** [`rsi/workflow.py`; RUN `python -m labs.lab_10_13_workflow_rsiflow`, `python -m labs.lab_10_14_workflow_selfhpo`] "An adaptive workflow — a bandit over operators plus bottleneck routing — and Self-HPO, where the improver spends 30% of its budget tuning itself. Both within noise here."
- **6:50** [README 'Where is the recursive part?' and 'Result'] "Honest summary: all methods improve the rule a lot; none beats the textbook rule; none clearly beats hill climbing at full budget; SO-RSI is the most sample-efficient and stable. And nothing here rewrites its own code — that's the ceiling of this demo."
- **7:20** [RUN `python -m pytest tests -q`] "33 tests cover the simulator, the whitelist, the budget, and each mechanism."
