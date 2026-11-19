"""Demo for video 1 — print the AIDE-style search tree as it grows.
    python -m labs.demo_aide_tree"""
from rsi.methods.aide_tree import aide_tree

from ._util import best_of, header, held_out, run, table

if __name__ == "__main__":
    header("demo", "AIDE-style tree search", "Which operator produced each node, and from which parent?")
    env, nodes = run(aide_tree, seed=1, budget=1200)
    table([[n.id, "—" if n.parent is None else n.parent, n.op, f"{n.score:.3f}" if n.result.ok else "CRASH", n.rule[:70]] for n in nodes[:24]], ["node", "parent", "operator", "train cost", "rule"])
    b = best_of(nodes)
    print(f"\n{len(nodes)} nodes · {sum(n.op == 'draft' for n in nodes)} drafts · {sum(n.op == 'debug' for n in nodes)} debugs · {sum(not n.result.ok for n in nodes)} crashed · {env.spent} shifts")
    print(f"best: node {b.id}, train {b.score:.3f} → held-out {held_out(b.rule)}\n  {b.rule}")
