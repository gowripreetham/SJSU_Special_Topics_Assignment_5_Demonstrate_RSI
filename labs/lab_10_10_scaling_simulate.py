"""Lab 10.10 — Scaling RSI, simulate campaigns: how often does the leader's review change what workers do?
    python -m labs.lab_10_10_scaling_simulate"""
from rsi import swarm

from ._util import BUDGET, DEV_SEEDS, header, held_out, mean, table

if __name__ == "__main__":
    header("10.10", "campaign log", "How many adoptions per campaign would mean the team has collapsed onto one line of search?")
    rows = []
    for campaign in (3, 6, 12):
        logs = [swarm.coordinated(BUDGET, s, campaign=campaign) for s in DEV_SEEDS]
        rows.append([campaign, f"{mean(l[2]['campaigns'] for l in logs):.1f}", f"{mean(l[2]['adoptions'] for l in logs):.1f}", f"{mean(l[2]['skipped_known_failures'] for l in logs):.1f}",
                     f"{mean(held_out(l[0][0].rule)['validation'] for l in logs):.3f}"])
    table(rows, ["steps per campaign", "campaigns", "adoptions", "known failures skipped", "validation cost"])
    print("\nFrequent review = more sharing and less independent exploration; rare review = three nearly independent runs.")
