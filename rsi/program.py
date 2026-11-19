"""Programs = one-line arithmetic expressions. This module makes them SAFE to run and easy to edit.

  compile_rule(src)   parse → check every AST node against a whitelist → compile. No names except the 8 variables and
                      6 math functions, no attributes, no subscripts, no calls to anything else. So an LLM-written (or
                      mutated) rule cannot import, read files or loop forever: it is pure arithmetic.
  edit operators      small source-to-source edits (the "code edits" the RSI methods choose between)
  edit_features       an 8-number description of an edit, computed WITHOUT running it (used by Budget RSI)
"""
from __future__ import annotations
import ast
import math
import random

from .problem import VARIABLES

FUNCS = {"exp": lambda x: math.exp(min(x, 50.0)), "log1p": lambda x: math.log1p(max(x, -0.999999)), "sqrt": lambda x: math.sqrt(max(x, 0.0)),
         "max": max, "min": min, "abs": abs}
ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)   # no ** : integer powers can be made arbitrarily expensive
MAX_NODES = 60


class RuleError(ValueError):
    pass


def parse(src: str) -> ast.Expression:
    if len(src) > 400:
        raise RuleError("rule is too long")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise RuleError(f"syntax error: {e.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise RuleError(f"rule is too large ({len(nodes)} nodes, limit {MAX_NODES})")
    for n in nodes:
        if not isinstance(n, ALLOWED):
            raise RuleError(f"{type(n).__name__} is not allowed in a rule")
        if isinstance(n, ast.Name) and n.id not in VARIABLES and n.id not in FUNCS:
            raise RuleError(f"unknown name '{n.id}'")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords):
            raise RuleError("only exp, log1p, sqrt, max, min, abs may be called")
        if isinstance(n, ast.Constant) and not (isinstance(n.value, (int, float)) and not isinstance(n.value, bool)):
            raise RuleError("only numeric constants are allowed")
    return tree


def compile_rule(src: str):
    """Source → callable(p, w, d, t, slack, wait, p_avg, n). Raises RuleError for anything outside the whitelist."""
    code = compile(parse(src), "<rule>", "eval")
    env = {"__builtins__": {}, **FUNCS}
    return _make(code, env)


def _make(code, env):
    def rule(p, w, d, t, slack, wait, p_avg, n):
        return eval(code, env, {"p": p, "w": w, "d": d, "t": t, "slack": slack, "wait": wait, "p_avg": p_avg, "n": n})   # noqa: S307 — AST-whitelisted arithmetic only
    return rule


def normalise(src: str) -> str:
    return ast.unparse(parse(src))


def variables_used(src: str) -> set[str]:
    return {n.id for n in ast.walk(parse(src)) if isinstance(n, ast.Name) and n.id in VARIABLES}


def constants(src: str) -> list[float]:
    return [float(n.value) for n in ast.walk(parse(src)) if isinstance(n, ast.Constant)]


# ------------------------------------------------------------------ edit operators
TERMS = ["w", "p", "slack", "wait", "d", "t", "n", "p_avg", "w / p", "max(0, slack)", "max(0, -slack)", "wait / p_avg", "slack / p_avg", "log1p(wait)", "sqrt(p)"]


def _const(rng: random.Random) -> float:
    return round(rng.choice([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]) * rng.uniform(0.7, 1.4), 3)


def op_draft(_parent: str, rng: random.Random) -> str:
    """A fresh small rule, unrelated to the parent."""
    a, b = rng.sample(TERMS, 2)
    return rng.choice([f"{a}", f"-({a})", f"{a} / ({b} + 1)", f"{a} - {_const(rng)} * {b}", f"{a} * {b}"])


def op_tweak_const(parent: str, rng: random.Random) -> str:
    tree = parse(parent)
    consts = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value not in (0, 1)]
    if not consts:
        return f"({parent}) * {_const(rng)} + {rng.choice(TERMS)}"
    c = rng.choice(consts)
    c.value = round(float(c.value) * rng.choice([0.5, 0.7, 0.85, 1.2, 1.5, 2.0]), 4)
    return ast.unparse(tree)


def op_add_term(parent: str, rng: random.Random) -> str:
    return f"({parent}) {rng.choice('+-')} {_const(rng)} * {rng.choice(TERMS)}"


def op_mul_term(parent: str, rng: random.Random) -> str:
    return f"({parent}) {rng.choice('*/')} ({rng.choice(TERMS)} + {_const(rng)})"


def op_wrap_urgency(parent: str, rng: random.Random) -> str:
    """Discount a rule by how far the order is from being late — the idea behind look-ahead rules."""
    return f"({parent}) * exp(-max(0, slack) / ({_const(rng)} * p_avg))"


def op_swap_var(parent: str, rng: random.Random) -> str:
    tree = parse(parent)
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in VARIABLES]
    if not names:
        return op_add_term(parent, rng)
    rng.choice(names).id = rng.choice(VARIABLES)
    return ast.unparse(tree)


def op_drop_term(parent: str, rng: random.Random) -> str:
    tree = parse(parent)
    bins = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
    if not bins:
        return parent
    b = rng.choice(bins)
    keep = rng.choice([b.left, b.right])
    return ast.unparse(_Replace(b, keep).visit(tree))


def introductions(parent: str, rng: random.Random, var: str) -> list[str]:
    """Targeted edits used by SO-RSI: ways to bring a variable the rule ignores into the rule (additive and multiplicative forms)."""
    c = _const(rng)
    forms = {"w": [f"({parent}) + {c} * w", f"({parent}) + {c} * w / p_avg", f"({parent}) * w"],
             "p": [f"({parent}) - {c} * p", f"({parent}) + {c} * p_avg / p", f"({parent}) / p"],
             "slack": [f"({parent}) - {c} * max(0, slack)", f"({parent}) * exp(-max(0, slack) / ({c} * p_avg))"],
             "wait": [f"({parent}) + {c} * wait / p_avg"]}
    return forms.get(var, [f"({parent}) + {c} * {var}"])


def op_debug(parent: str, _rng: random.Random) -> str:
    """Repair the usual crash cause: division by something that can be zero."""
    tree = parse(parent)
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            n.right = ast.BinOp(ast.Call(ast.Name("abs", ast.Load()), [n.right], []), ast.Add(), ast.Constant(0.001))
    return ast.unparse(ast.fix_missing_locations(tree))


class _Replace(ast.NodeTransformer):
    def __init__(self, target, new):
        self.target, self.new = target, new

    def visit(self, node):
        return self.new if node is self.target else super().visit(node)


OPERATORS = {"draft": op_draft, "tweak_const": op_tweak_const, "add_term": op_add_term, "mul_term": op_mul_term, "wrap_urgency": op_wrap_urgency,
             "swap_var": op_swap_var, "drop_term": op_drop_term}
IMPROVE_OPS = [o for o in OPERATORS if o != "draft"]


def apply(op: str, parent: str, rng: random.Random) -> str:
    """Apply an operator; the result is normalised source, or the raw text if it does not even parse (a real failed edit)."""
    child = OPERATORS[op](parent, rng) if op in OPERATORS else op_debug(parent, rng)
    try:
        return normalise(child)
    except RuleError:
        return child


def edit_features(parent: str, child: str, op: str) -> list[float]:
    """8 coordinates describing an edit, computed from the two sources alone (no execution)."""
    try:
        pv, cv, pc, cc = variables_used(parent), variables_used(child), constants(parent), constants(child)
        size_p, size_c = len(list(ast.walk(parse(parent)))), len(list(ast.walk(parse(child))))
    except RuleError:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return [(size_c - size_p) / 10.0,                          # 1 growth in size
            float(len(cv - pv)),                               # 2 variables introduced
            float(len(pv - cv)),                               # 3 variables removed
            float("w" in cv and "p" in cv),                    # 4 uses both weight and prep time
            float("slack" in cv or "d" in cv),                 # 5 is due-date aware
            float("exp" in child) - float("exp" in parent),    # 6 adds / removes an exponential
            float(len(cc) - len(pc)),                          # 7 constants added
            float(size_c > 45)]                                # 8 near the size limit
