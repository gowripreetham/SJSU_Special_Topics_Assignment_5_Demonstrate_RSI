# Results — 20 seeds × 2400 simulated shifts per run (100 full evaluations)

Relative cost = total weighted tardiness ÷ FIFO's on the same shifts (lower is better). Mean ± sd over seeds. Rule reported per run = best by train score.

## Hand-designed reference rules (held-out test)

| Rule | test | shifted day |
|---|---|---|
| FIFO | 1.000 | 1.000 |
| EDD | 0.609 | 0.657 |
| WSPT | 0.569 | 0.569 |
| ATC | 0.457 | 0.503 |
| ATC (k tuned on train) | 0.457 | 0.503 |

## Methods at equal budget

| Method | train | validation | **test** (mean ± sd) | test median | shifted day | RSI-Index (median run) | runs beating tuned ATC | runs whose rule crashes on test | rules evaluated | crashed in search |
|---|---|---|---|---|---|---|---|---|---|---|
| Random search | 0.514 ± 0.034 | 0.506 ± 0.044 | **0.552 ± 0.049** | 0.569 | 0.566 ± 0.036 | 0.26 | 0/20 | 0/20 | 111 | 1 |
| Naive RSI (hill climb) | 0.494 ± 0.031 | 0.474 ± 0.041 | **0.512 ± 0.041** | 0.494 | 0.552 ± 0.038 | 0.75 | 0/20 | 0/20 | 129 | 19 |
| AIDE-style tree search | 0.502 ± 0.033 | 0.480 ± 0.044 | **0.523 ± 0.039** | 0.524 | 0.560 ± 0.036 | 0.56 | 0/20 | 0/20 | 121 | 11 |
| Budget RSI (readout) | 0.532 ± 0.049 | 0.514 ± 0.055 | **1.015 ± 2.113** | 0.545 | 0.587 ± 0.053 | 0.42 | 0/20 | 1/20 | 101 | 1 |
| Budget RSI — no edit features | 0.487 ± 0.034 | 0.466 ± 0.045 | **0.502 ± 0.040** | 0.493 | 0.539 ± 0.036 | 0.76 | 0/20 | 0/20 | 110 | 10 |
| SO-RSI (diagnostic) | 0.486 ± 0.024 | 0.463 ± 0.033 | **0.501 ± 0.030** | 0.492 | 0.543 ± 0.031 | 0.77 | 0/20 | 0/20 | 115 | 19 |

A rule that divides by something that is zero on an unseen shift crashes there and is scored 9.99 — that is what inflates a mean. Medians are robust to it.

## Paired differences in test cost (same seeds; negative = lower cost; 95% bootstrap interval)

| Comparison | mean difference | 95% interval | reading |
|---|---|---|---|
| Random search − naive RSI | +0.040 | [+0.012, +0.069] | worse |
| AIDE-style tree search − naive RSI | +0.012 | [-0.008, +0.033] | no clear difference |
| Budget RSI (readout) − naive RSI | +0.504 | [+0.013, +1.451] | worse |
| Budget RSI — no edit features − naive RSI | -0.010 | [-0.032, +0.014] | no clear difference |
| SO-RSI (diagnostic) − naive RSI | -0.010 | [-0.030, +0.010] | no clear difference |
| Budget RSI: readout vs no-features | +0.513 | [+0.021, +1.468] | worse |

## Half-way through the budget (1,200 shifts) — exploratory, added after looking at the learning curves

| Comparison | mean difference | 95% interval | reading |
|---|---|---|---|
| AIDE-style tree search − naive RSI | +0.009 | [-0.016, +0.034] | no clear difference |
| Budget RSI (readout) − naive RSI | +0.008 | [-0.022, +0.039] | no clear difference |
| SO-RSI (diagnostic) − naive RSI | -0.034 | [-0.055, -0.012] | better |

## Behaviour: anomalies per 100 steps

| Loop | recurrence | opposing edits | expectation mismatch | diagnoses per run |
|---|---|---|---|---|
| naive (monitor only) | 15.0 | 3.9 | 12.8 | 0.0 |
| SO-RSI | 12.2 | 3.6 | 11.2 | 8.1 |

## Budget RSI — does the readout rank edits? (lab 10.04)

Every proposed edit labelled, 60 edits × 10 seeds, fit on the first 60%, measured on the rest.

- Spearman rank correlation, predicted vs actual child cost: **0.36 ± 0.17**
- Share of edits that improve on their parent: 20% in the readout's top quarter vs 18% overall
- Share of edits that crash: 0% in the readout's top quarter vs 2% overall

## Budget RSI — extension (lab 10.05), separate seeds 200+

| Variant | test mean | test median | runs whose rule crashes on test |
|---|---|---|---|
| readout picks the top edit | 0.545 | 0.544 | 0/20 |
| readout only filters out the worst half | 0.516 | 0.505 | 0/20 |
| no edit features (random sibling) | 0.511 | 0.490 | 0/20 |

## What budget buys — median test cost, 10 seeds (seeds 300+)

| Shifts | Random search | Naive RSI (hill climb) | AIDE-style tree search | Budget RSI (readout) | SO-RSI (diagnostic) |
|---|---|---|---|---|---|
| 600 | 0.608 | 0.594 | 0.534 | 0.590 | 0.516 |
| 2400 | 0.530 | 0.527 | 0.469 | 0.545 | 0.487 |
| 6000 | 0.500 | 0.478 | 0.470 | 0.495 | 0.472 |

## Coordination, adaptive workflow, self-tuning (same budget and seeds)

| Variant | test | RSI-Index |
|---|---|---|
| Swarm — 3 independent workers | 0.526 ± 0.051 | 0.55 |
| Swarm — 3 coordinated workers | 0.510 ± 0.038 | 0.65 |
| Adaptive workflow (meta-agent) | 0.498 ± 0.039 | 0.73 |
| Tree search + Self-HPO | 0.531 ± 0.048 | 0.51 |

| Comparison | mean difference | 95% interval | reading |
|---|---|---|---|
| coordinated − independent | -0.016 | [-0.044, +0.011] | no clear difference |
| self-HPO − default tree search | +0.008 | [-0.015, +0.032] | no clear difference |
| adaptive workflow − naive | -0.013 | [-0.034, +0.007] | no clear difference |

## Best rule found (chosen by validation score across all runs)

`w / p * exp(-max(0, n) / (1.646 * p_avg)) * exp(-max(0, slack) / (2.36 * p_avg)) + 4.115 * n`

validation 0.416 · test 0.463 · shifted day 0.507
