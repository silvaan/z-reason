"""Variable tracking in natural language.

The same chains as `tasks.py` (shuffled facts, one chain of exactly the requested depth,
many distractor chains) rendered as sentences about people and their numbers:

    Ruth has the number 4. Omar has the same number as Ruth. Kofi was given whatever
    number Omar has. Lena's number is 7. ...

Every fact and question is drawn from a set of templates. The last few templates of each
set are held out (`split="test"` uses only those), so evaluation can separate reasoning
depth from the wording seen in training.

Questions follow Jev's schema and Laya's sequence layout (one sequence per question):

* choice  "What number does X have?"            10 options, 0-9
* score   "How large is X's number?"             5 ordered levels
* noul    "Does X's number come from Y's, ..."   true / false
"""

from __future__ import annotations

import random

from .tasks import Example, _ancestors, _build_program

PEOPLE = [
    "Ruth", "Omar", "Kofi", "Lena", "Igor", "Maya", "Hugo", "Nina", "Raul", "Tara", "Ivan", "Zoe",
    "Abel", "Cleo", "Dino", "Erin", "Felix", "Gina", "Hans", "Iris", "Jack", "Kira", "Leon", "Mona",
    "Nash", "Olga", "Paul", "Rosa", "Saul", "Tina", "Uma", "Vera", "Walt", "Yara", "Ana", "Ben",
    "Carl", "Dana", "Ella", "Finn", "Gus", "Hana", "Ines", "Joel", "Kurt", "Lily", "Max", "Nora",
    "Otto", "Pia", "Rex", "Sara", "Theo", "Ugo", "Vito", "Will",
]

# The last HELD_OUT[...] templates of each list are held out for split="test". Fact templates
# name the receiver first or the giver first, so word order alone does not give the direction.
ROOT = [
    "{x} has the number {v}.",
    "{x}'s number is {v}.",
    "The number of {x} is {v}.",
    "{x} was given the number {v}.",
    "{x} picked the number {v}.",
    "The number {v} belongs to {x}.",
    "{x} starts with the number {v}.",
    "{x} holds the number {v}.",  # held out
    "The number {v} was assigned to {x}.",  # held out
]
COPY = [
    "{x} has the same number as {y}.",
    "{x}'s number equals {y}'s.",
    "{x} copied the number of {y}.",
    "{x} was given whatever number {y} has.",
    "{x} took {y}'s number.",
    "{x} uses the number that {y} has.",
    "{x} borrowed the number of {y}.",
    "{y} gave their number to {x}.",
    "{y}'s number was copied by {x}.",
    "{y}'s number was handed to {x}.",
    "The number of {x} matches that of {y}.",  # held out
    "{x} adopted the number of {y}.",  # held out
    "{y} passed their number on to {x}.",  # held out
]
CHOICE_Q = [
    "What number does {x} have?",
    "Which number belongs to {x}?",
    "What is {x}'s number?",
    "Which number is {x} holding?",
    "Tell me the number of {x}.",  # held out
    "What number did {x} end up with?",  # held out
]
SCORE_Q = [
    "How large is {x}'s number?",
    "How big is the number of {x}?",
    "How high is {x}'s number?",
    "Is {x}'s number small or large?",
    "Rate the size of {x}'s number.",  # held out
    "How large a number does {x} have?",  # held out
]
NOUL_Q = [
    "Does {x}'s number come from {y}'s, directly or through others?",
    "Is {x}'s number taken, directly or indirectly, from {y}?",
    "Does {x} get its number from {y}, possibly through other people?",
    "Can {x}'s number be traced back to {y}?",
    "Did {y}'s number reach {x}, directly or through a chain of copies?",  # held out
    "Is {y} a source of {x}'s number?",  # held out
]
HELD_OUT = {"ROOT": 2, "COPY": 3, "CHOICE_Q": 2, "SCORE_Q": 2, "NOUL_Q": 2}

CHOICE_CRITERIA = [str(d) for d in range(10)]
SCORE_CRITERIA = ["0 or 1", "2 or 3", "4 or 5", "6 or 7", "8 or 9"]


def templates(name: str, split: str) -> list[str]:
    t, k = globals()[name], HELD_OUT[name]
    return t[-k:] if split == "test" else t[:-k]


def _pick(rng: random.Random, name: str, split: str) -> str:
    return rng.choice(templates(name, split))


def make_nl_example(
    rng: random.Random,
    depth: int,
    n_vars: int | None = None,
    n_questions: int = 2,
    min_vars: int = 16,
    max_vars: int = 28,
    types: tuple[str, ...] = ("choice", "score", "noul"),
    split: str = "train",
) -> Example:
    if n_vars is None:
        n_vars = rng.randint(max(min_vars, depth + 1), max(max_vars, depth + 1))
    letters, parent, dep, value, _ = _build_program(rng, depth, n_vars)
    who = dict(zip(letters, rng.sample(PEOPLE, n_vars)))
    facts = [_pick(rng, "ROOT", split).format(x=who[v], v=value[v]) if parent[v] is None
             else _pick(rng, "COPY", split).format(x=who[v], y=who[parent[v]]) for v in letters]
    rng.shuffle(facts)
    facts = [f[0].upper() + f[1:] for f in facts]

    questions, labels, meta = {}, {}, {}
    for qi in range(n_questions):
        x = letters[depth] if qi == 0 else rng.choice(letters)
        qtype = rng.choice(types)
        qid = f"q{qi}"
        if qtype == "choice":
            questions[qid] = {"type": "choice", "instructions": _pick(rng, "CHOICE_Q", split).format(x=who[x]),
                              "criteria": CHOICE_CRITERIA}
            labels[qid] = value[x]
        elif qtype == "score":
            questions[qid] = {"type": "score", "instructions": _pick(rng, "SCORE_Q", split).format(x=who[x]),
                              "criteria": SCORE_CRITERIA}
            labels[qid] = value[x] // 2
        else:
            anc = _ancestors(parent, x)
            others = [u for u in letters if u != x and u not in anc]
            if anc and (rng.random() < 0.5 or not others):
                y, lab = rng.choice(anc), 1
            else:
                y, lab = rng.choice(others), 0
            questions[qid] = {"type": "noul", "instructions": _pick(rng, "NOUL_Q", split).format(x=who[x], y=who[y])}
            labels[qid] = lab
        meta[qid] = {"depth": dep[x]}
    return Example(state=" ".join(facts), questions=questions, labels=labels, meta=meta)


def make_nl_dataset(seed: int, n: int, depths, **kw) -> list[Example]:
    rng = random.Random(seed)
    return [make_nl_example(rng, d, **kw) for d in depths for _ in range(n)]
