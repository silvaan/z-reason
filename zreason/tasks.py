"""Synthetic reasoning tasks with controllable difficulty.

Variable-tracking task (in the spirit of RULER's variable tracking)
-------------------------------------------------------------------
The state is a shuffled program made of independent assignment chains, e.g.

    q=4;m=q;Z=m;c=7;k=c;w=2;T=w;...     (default: values are copied along each chain)
    q=4;m=q+7;Z=m-2;...                 (ops=True: plus arithmetic modulo 10)

Answering a question about variable ``x`` requires following its chain up to
the root: ``depth(x)`` sequential hops. Lines are shuffled, so there is no
left-to-right shortcut. Depth is the difficulty knob.

Design choices that remove shortcuts:
* one chain has exactly the requested depth, the others have random lengths up
  to that depth, and the program always holds 24-40 variables. Many chains means
  many roots, so guessing among root values is close to chance, and the
  sequence length does not reveal the difficulty (train and test lengths match).
* chains do not branch, so no chain is "bigger" in a way that hints at the answer.

Every state carries several independent, typed questions (Jev-style):

* choice  ``val x``    -> value of x, 10 options
* score   ``lvl x``    -> ordinal bucket of the value: 0-1 < 2-3 < ... < 8-9
* noul    ``dep x y``  -> probability that x (transitively) depends on y
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

NAMES = string.ascii_lowercase + string.ascii_uppercase  # 52 single-character variables
CHOICE_CRITERIA = {str(d): str(d) for d in range(10)}
SCORE_CRITERIA = ["0-1", "2-3", "4-5", "6-7", "8-9"]


@dataclass
class Example:
    state: str
    questions: dict[str, dict]            # Jev-style question specs
    labels: dict[str, int]                # gold label (option index / level / 0-1)
    meta: dict[str, dict] = field(default_factory=dict)  # e.g. {"depth": 5}


def _offset(rng: random.Random, ops: bool) -> int:
    return rng.choice([-1, 1]) * rng.randint(1, 9) if ops else 0


def _build_program(rng: random.Random, depth: int, n_vars: int, ops: bool = False):
    """Returns names (chain order), parent, depth, value, program text."""
    assert n_vars >= depth + 1, "need at least depth+1 variables"
    names = rng.sample(NAMES, n_vars)
    lengths = [depth + 1]
    left = n_vars - depth - 1
    while left > 0:
        lengths.append(rng.randint(1, min(depth + 1, left)))
        left -= lengths[-1]

    parent: dict[str, str | None] = {}
    op: dict[str, int] = {}
    dep: dict[str, int] = {}
    i = 0
    for n in lengths:
        chain = names[i: i + n]
        i += n
        parent[chain[0]], op[chain[0]], dep[chain[0]] = None, rng.randrange(10), 0
        for k in range(1, n):
            parent[chain[k]], op[chain[k]], dep[chain[k]] = chain[k - 1], _offset(rng, ops), k

    value: dict[str, int] = {}
    for v in names:  # chain order is topological
        value[v] = op[v] % 10 if parent[v] is None else (value[parent[v]] + op[v]) % 10

    lines = []
    for v in names:
        if parent[v] is None:
            lines.append(f"{v}={op[v]}")
        elif op[v] == 0:
            lines.append(f"{v}={parent[v]}")
        else:
            sign = "+" if op[v] > 0 else "-"
            lines.append(f"{v}={parent[v]}{sign}{abs(op[v])}")
    rng.shuffle(lines)
    return names, parent, dep, value, ";".join(lines)


def _ancestors(parent, v):
    out = []
    while parent[v] is not None:
        v = parent[v]
        out.append(v)
    return out


def make_chain_example(
    rng: random.Random,
    depth: int,
    n_vars: int | None = None,
    n_questions: int = 4,
    min_vars: int = 24,
    max_vars: int = 40,
    types: tuple[str, ...] = ("choice", "score", "noul"),
    ops: bool = False,
) -> Example:
    if n_vars is None:
        n_vars = rng.randint(max(min_vars, depth + 1), max(max_vars, depth + 1))
    names, parent, dep, value, state = _build_program(rng, depth, n_vars, ops)
    main_leaf = names[depth]

    questions, labels, meta = {}, {}, {}
    for qi in range(n_questions):
        x = main_leaf if qi == 0 else rng.choice(names)
        qtype = rng.choice(types)
        qid = f"q{qi}"
        if qtype == "choice":
            questions[qid] = {"type": "choice", "instructions": f"val {x}", "criteria": CHOICE_CRITERIA}
            labels[qid] = value[x]
        elif qtype == "score":
            questions[qid] = {"type": "score", "instructions": f"lvl {x}", "criteria": SCORE_CRITERIA}
            labels[qid] = value[x] // 2
        else:
            anc = _ancestors(parent, x)
            others = [u for u in names if u != x and u not in anc]
            if anc and (rng.random() < 0.5 or not others):
                y, lab = rng.choice(anc), 1
            else:
                y, lab = rng.choice(others), 0
            questions[qid] = {"type": "noul", "instructions": f"dep {x} {y}"}
            labels[qid] = lab
        meta[qid] = {"depth": dep[x]}
    return Example(state=state, questions=questions, labels=labels, meta=meta)


def make_dataset(seed: int, n: int, depths, **kw) -> list[Example]:
    """Fixed evaluation set: `n` examples per depth in `depths`."""
    rng = random.Random(seed)
    return [make_chain_example(rng, d, **kw) for d in depths for _ in range(n)]
