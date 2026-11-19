# Codelabs — course section 10, rebuilt on the pharmacy-queue problem

Run any lab with `python -m labs.<name>` from the repository root. Each prints a question to answer **before** reading its output.
Labs use development seeds (fast); the reported study is `python -m rsi.experiments`.

| Course group | Lab | Script | What it shows |
|---|---|---|---|
| 00 Read a frontier claim | 10.01 | `lab_10_01_claim_vs_evidence` | source claim vs my adaptation vs my observation, per method |
| | 10.02 | `lab_10_02_audit` | audit of this repo's own study |
| 01 Budget RSI | 10.03 | `lab_10_03_budget_run_demo` | readout chooses which edit is evaluated |
| | 10.04 | `lab_10_04_budget_analyze_mechanism` | out-of-sample quality of the readout, learned weights |
| | 10.05 | `lab_10_05_budget_extend` | extension: filter instead of pick |
| 02 SO-RSI | 10.06 | `lab_10_06_so_run_demo` | anomaly → probes → memory → targeted edits |
| | 10.07 | `lab_10_07_so_analyze_probes` | probes on rules with planted faults |
| | 10.08 | `lab_10_08_so_apply` | thrash rate and score, first- vs second-order |
| 03 Scaling & coordination | 10.09 | `lab_10_09_scaling_compare` | independent vs coordinated workers |
| | 10.10 | `lab_10_10_scaling_simulate` | campaign length vs sharing |
| | 10.11 | `lab_10_11_scaling_swarm` | swarm size at a fixed budget |
| 04 Adaptive workflows | 10.12 | `lab_10_12_workflow_compare` | fixed vs adaptive loop |
| | 10.13 | `lab_10_13_workflow_rsiflow` | what the meta-agent learned |
| | 10.14 | `lab_10_14_workflow_selfhpo` | the improver tunes itself |
| 05 Environments & metrics | 10.15 | `lab_10_15_metrics_rsi_index` | RSI-Index against a hand-designed reference |
| | 10.16 | `lab_10_16_metrics_environment` | budget, cache, crashes, whitelist |
| | 10.17 | `lab_10_17_metrics_report` | the full results table |
