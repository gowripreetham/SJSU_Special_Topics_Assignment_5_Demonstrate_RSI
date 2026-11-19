"""Shared pieces: where every method starts, how edits are proposed, and the journal entry for one evaluated rule."""
from __future__ import annotations
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field

from .. import program
from ..env import Env, Result

START = "-d"      # every method starts from the same rule: earliest due date (EDD)
FULL = 24         # shifts in one full evaluation (the train split)


@dataclass
class Node:
    id: int
    rule: str
    op: str
    parent: int | None
    result: Result
    meta: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.result.score


class GrammarProposer:
    """Offline proposer: applies a source-to-source edit operator. Stochastic, cheap, and often wrong — like any proposer."""
    name = "grammar"

    def edit(self, op: str, parent: str, rng: random.Random, hint: str = "") -> str:
        return program.apply(op, parent, rng)


class LLMProposer:
    """Optional: an LLM writes the edit (OpenRouter). The reply is treated as untrusted text — it still has to pass the
    AST whitelist in program.compile_rule, and its score still comes from the environment."""
    name = "llm"

    def __init__(self, chat=None, model: str | None = None):
        self.model = model or os.environ.get("RSI_MODEL", "openrouter/auto")
        self.chat = chat or self._openrouter

    def _openrouter(self, prompt: str) -> str:
        req = urllib.request.Request(os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions", method="POST",
                                     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
                                     data=json.dumps({"model": self.model, "temperature": 0.9, "messages": [{"role": "user", "content": prompt}]}).encode())
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def edit(self, op: str, parent: str, rng: random.Random, hint: str = "") -> str:
        prompt = ("You improve a dispatch rule for a hospital pharmacy compounding queue (minimise total weighted tardiness). The rule is ONE Python arithmetic "
                  "expression; the waiting order with the highest value is prepared next. Variables: p (prep minutes), w (clinical weight), d (due time), t (now), "
                  "slack (= d - t - p), wait (minutes waited), p_avg (mean prep time of waiting orders), n (orders waiting). Functions: exp, log1p, sqrt, max, min, abs. "
                  "Operators: + - * /. No other names, no **, no conditionals.\n"
                  f"Current rule: {parent}\nKind of edit requested: {op}. {hint}\nReply with the new expression only, on one line.")
        try:
            lines = [l.strip().strip("`") for l in self.chat(prompt).strip().splitlines() if l.strip() and not l.strip().startswith("```")]
            text = lines[-1].split("=", 1)[-1].strip() if lines else parent
            return program.normalise(text)
        except program.RuleError:
            return text                       # an invalid edit is a real outcome: the environment will reject it
        except Exception:                     # network / auth problem: fall back to the offline operator
            return program.apply(op, parent, rng)


def default_proposer():
    return LLMProposer() if os.environ.get("RSI_LLM") == "1" and os.environ.get("OPENROUTER_API_KEY") else GrammarProposer()


def best_of(nodes: list[Node]) -> Node:
    return min(nodes, key=lambda n: n.score)


def evaluate_node(env: Env, nodes: list[Node], rule: str, op: str, parent: int | None, **meta) -> Node:
    node = Node(len(nodes), rule, op, parent, env.evaluate(rule), meta)
    nodes.append(node)
    if node.result.ok:
        env.note_best(best_of(nodes).score, best_of(nodes).rule)
    return node
