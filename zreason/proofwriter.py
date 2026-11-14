"""ProofWriter (Tafjord et al., 2021), open-world assumption, as typed decisions.

Each theory is a short text of facts and rules ("Gary is cold. If something is cold then it
is big. ..."); each question is a statement whose answer is True, False or Unknown, labelled
with the depth of its proof (QDep, 0-5). The standard protocol trains on the depth-3 set and
tests on the depth-5 set, so depths 4 and 5 are deeper than anything seen in training.
NatLang holds crowd-sourced paraphrases and birds-electricity a different domain.

Data: huggingface.co/datasets/hitachi-nlp/proofwriter_processed_OWA (CC BY 4.0), downloaded to
cache/proofwriter by `download()`.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .tasks import Example

ROOT = Path(__file__).resolve().parent.parent / "cache" / "proofwriter"
REPO = "hitachi-nlp/proofwriter_processed_OWA"
ANSWERS = {"True": "True", "False": "False", "Unknown": "Unknown"}
LABEL = {a: i for i, a in enumerate(ANSWERS)}


def download():
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", local_dir=ROOT,
                      allow_patterns=["README.md", "depth-3/*", "depth-5/*", "birds-electricity/*", "NatLang/*"])


def sentences(theory: str) -> list[str]:
    # ';' would split a sentence into two facts in batching.state_facts (a few NatLang theories use it)
    return [s.strip() for s in re.findall(r"[^.]+\.", theory.replace(";", ","))]


def load(config: str, split: str) -> list[Example]:
    """One example per theory; the state has one sentence per line, so batching splits it into facts."""
    path = ROOT / config / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        download()
    out = []
    for _, r in pd.read_parquet(path).iterrows():
        qs = {k: q for k, q in r["questions"].items() if q is not None}
        out.append(Example(
            state="\n".join(sentences(r["theory"])),
            questions={k: {"type": "choice", "instructions": q["question"], "criteria": ANSWERS} for k, q in qs.items()},
            labels={k: LABEL[q["answer"]] for k, q in qs.items()},
            meta={k: {"depth": int(q["QDep"])} for k, q in qs.items()},
        ))
    return out


# Training augmentation: rename the entities, attributes and relations of a theory with a
# consistent bijection, so the 4.8k depth-3 training theories cannot be memorized (their
# vocabulary is 8 names, 10 animals, 14 adjectives and 6 verbs). Logic and labels are unchanged.
# Each pool holds the original words, and the pools are disjoint from each other and from the
# function words of the theories.
NAMES = ["Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry", "Ruth", "Omar", "Kofi", "Lena",
         "Igor", "Maya", "Hugo", "Nina", "Raul", "Tara", "Ivan", "Zoe", "Abel", "Cleo", "Dino", "Gina", "Hans",
         "Iris", "Jack", "Kira", "Leon", "Mona", "Nash", "Olga", "Paul", "Rosa", "Saul", "Tina", "Uma", "Vera"]
ANIMALS = ["bald eagle", "bear", "cat", "cow", "dog", "lion", "mouse", "rabbit", "squirrel", "tiger", "horse",
           "goat", "fox", "wolf", "owl", "duck", "sheep", "deer", "frog", "snake", "whale", "shark", "crab",
           "panda", "koala", "zebra", "camel", "moose", "otter", "hawk"]
ADJECTIVES = ["big", "blue", "cold", "furry", "green", "kind", "nice", "quiet", "red", "rough", "round", "smart",
              "white", "young", "tall", "short", "soft", "hard", "loud", "calm", "brave", "shy", "warm", "wet",
              "dry", "fast", "slow", "old", "happy", "sad", "strong", "weak", "clean", "dirty", "bright", "dark",
              "heavy", "sharp", "pink", "black", "gray", "shiny", "sleepy"]
VERBS = [("chase", "chases"), ("eat", "eats"), ("like", "likes"), ("need", "needs"), ("see", "sees"),
         ("visit", "visits"), ("help", "helps"), ("call", "calls"), ("follow", "follows"), ("hug", "hugs"),
         ("watch", "watches"), ("push", "pushes"), ("meet", "meets"), ("love", "loves"), ("fear", "fears"),
         ("trust", "trusts"), ("feed", "feeds"), ("kick", "kicks"), ("greet", "greets"), ("pull", "pulls")]


def rename(e: Example, rng: random.Random) -> Example:
    """The same theory and questions under a random bijective renaming of its vocabulary."""
    text = e.state + "\n" + "\n".join(q["instructions"] for q in e.questions.values())
    low = text.lower()
    mapping = {}
    for pool in (NAMES, ANIMALS, ADJECTIVES):
        used = [w for w in pool if re.search(rf"\b{w.lower()}\b", low)]
        for w, v in zip(used, rng.sample(pool, len(used))):
            mapping[w.lower()] = v.lower()
    used = [v for v in VERBS if re.search(rf"\b({v[0]}|{v[1]})\b", low)]
    for v, n in zip(used, rng.sample(VERBS, len(used))):
        mapping[v[0]], mapping[v[1]] = n
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, mapping), key=len, reverse=True)) + r")\b", re.I)

    def sub(t: str) -> str:
        def one(m):
            w = mapping[m[0].lower()]
            return w[0].upper() + w[1:] if m[0][0].isupper() else w
        return pat.sub(one, t)
    return replace(e, state=sub(e.state),
                   questions={k: {**q, "instructions": sub(q["instructions"])} for k, q in e.questions.items()})


def vector_strings() -> tuple[list[str], set[str]]:
    """Texts to encode for laya_features.build_vectors, and the theory sentences that set the
    standardization statistics."""
    strings = all_strings()
    stat = {s for e in load("depth-3", "train") for s in e.state.split("\n")}
    return strings, stat


def all_strings() -> list[str]:
    """Every sentence, question and option of the configurations we use."""
    out = set(ANSWERS)
    if not (ROOT / "depth-3").exists():
        download()
    for path in sorted(ROOT.glob("*/*.parquet")):
        for e in load(path.parent.name, path.name.split("-")[0]):
            out.update(e.state.split("\n"))
            out.update(q["instructions"] for q in e.questions.values())
    return sorted(out)
