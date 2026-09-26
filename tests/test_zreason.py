import random

import torch

from zreason.api import ZReason, confidence
from zreason.batching import collate
from zreason.model import ZConfig, ZReasoner
from zreason.tasks import _ancestors, _build_program, make_chain_example
from zreason.tokenizer import CharTokenizer

TOK = CharTokenizer()


def _eval_program(state: str) -> dict:
    defs = dict(line.split("=") for line in state.split(";"))
    memo = {}

    def val(v):
        if v not in memo:
            rhs = defs[v]
            if rhs.isdigit():
                memo[v] = int(rhs) % 10
            elif len(rhs) == 1:
                memo[v] = val(rhs)
            else:
                sign = 1 if rhs[1] == "+" else -1
                memo[v] = (val(rhs[0]) + sign * int(rhs[2:])) % 10
        return memo[v]

    return {v: val(v) for v in defs}


def test_labels_match_program_semantics():
    rng = random.Random(0)
    for i in range(400):
        ex = make_chain_example(rng, rng.randint(1, 16), ops=i % 2 == 0)
        vals = _eval_program(ex.state)
        for qid, q in ex.questions.items():
            x = q["instructions"].split()[1]
            if q["type"] == "choice":
                assert ex.labels[qid] == vals[x]
            elif q["type"] == "score":
                assert ex.labels[qid] == vals[x] // 2


def test_program_structure():
    rng = random.Random(1)
    for d in range(1, 17):
        for n in (d + 1, 24, 40):
            names, parent, dep, _, state = _build_program(rng, d, max(n, d + 1))
            assert max(dep.values()) == d and dep[names[d]] == d
            assert len(state.split(";")) == len(names) == len(set(names))
            for v in names:
                assert len(_ancestors(parent, v)) == dep[v]
                assert dep[v] <= d  # no chain deeper than the target


def test_tokenizer_roundtrip():
    s = "a=3;B=a;Z=B"
    assert TOK.decode(TOK.encode(s)) == s


def _tiny():
    torch.manual_seed(0)
    return ZReasoner(ZConfig(vocab_size=TOK.vocab_size, d=64, heads=4))


def test_questions_are_independent():
    """Adding/removing/reordering other questions must not change an answer."""
    m = _tiny().eval()
    state = "a=3;b=a+2;c=b-1"
    q1 = {"type": "choice", "instructions": "val c", "criteria": {str(i): str(i) for i in range(10)}}
    q2 = {"type": "noul", "instructions": "dep c a"}
    q3 = {"type": "score", "instructions": "lvl b", "criteria": ["0-1", "2-3", "4-5", "6-7", "8-9"]}
    b1 = collate([{"state": state, "questions": {"x": q1}}], TOK)
    b2 = collate([{"state": state, "questions": {"z": q3, "y": q2, "x": q1}}], TOK)
    # z_0 noise is drawn per slot, so compare with deterministic init
    with torch.no_grad():
        o1 = m(b1, steps=5, noise=0.0)
        o2 = m(b2, steps=5, noise=0.0)
    assert torch.allclose(o1["logits"][0], o2["logits"][2], atol=1e-5)


def test_api_output_schema():
    api = ZReason(_tiny(), device="cpu")
    res = api.predict("a=3;b=a+2", {
        "v": {"type": "choice", "instructions": "val b", "criteria": {str(i): str(i) for i in range(10)}},
        "s": {"type": "score", "instructions": "lvl b", "criteria": ["0-1", "2-3", "4-5", "6-7", "8-9"]},
        "n": {"type": "noul", "instructions": "dep b a"},
    }, steps=3, trace=True)
    a = res["answers"]
    assert set(a) == {"v", "s", "n"} and res["steps"] == 3 and len(res["trace"]) == 3
    assert a["v"]["choice"] in a["v"]["probabilities"]
    assert abs(sum(a["v"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["s"]["score"] <= 4 and set(a["s"]["legend"]) == {"0", "1", "2", "3", "4"}
    assert 0 <= a["n"]["noul"] <= 1 and "confidence" not in a["n"]
    auto = api.predict("a=3;b=a+2", {"n": {"type": "noul", "instructions": "dep b a"}}, steps="auto", max_steps=20)
    assert 1 <= auto["steps"] <= 20


def test_confidence():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert abs(confidence([1 / 3] * 3)) < 1e-9
