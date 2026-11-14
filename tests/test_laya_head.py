import random
import re

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from zreason.laya_head import QTYPES, ZLaya, add_facts, build_sequence, collate, fact_spans
from zreason.nl_tasks import COPY, ROOT, make_nl_example, templates


def _tiny_zlaya(untied_steps=0, units="tokens"):
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                           num_attention_heads=2, pad_token_id=0)
    m = ZLaya(ModernBertModel(cfg), untied_steps=untied_steps, units=units).eval()
    ids = torch.randint(1, 128, (3, 12))
    att = torch.ones_like(ids, dtype=torch.bool)
    att[2, 9:] = False
    b = {"input_ids": ids, "attention_mask": att, "qtype": torch.tensor([0, 1, 2]),
         "marker_pos": torch.tensor([[1, 2], [3, 4], [5, 6]]), "marker_mask": torch.ones(3, 2, dtype=torch.bool)}
    return m, b


def _eval_nl(state: str) -> dict:
    """Recover every person's number from the sentences alone."""
    def pattern(t):  # named groups: some templates name the giver first
        return re.compile(t.replace(".", "\\.").replace("{x}", "(?P<x>\\w+)").replace("{y}", "(?P<y>\\w+)")
                          .replace("{v}", "(?P<v>\\d)"))
    root, copy = [pattern(t) for t in ROOT], [pattern(t) for t in COPY]
    val, par = {}, {}
    for s in re.findall(r"[^.]+\.", state):
        s = s.strip()
        if m := next((m for r in root if (m := r.fullmatch(s))), None):
            val[m["x"]] = int(m["v"])
        else:
            m = next(m for r in copy if (m := r.fullmatch(s)))
            par[m["x"]] = m["y"]
    while par:
        for x, y in list(par.items()):
            if y in val:
                val[x] = val[y]
                del par[x]
    return val


def test_steps1_is_the_one_pass_head():
    """With one step the recurrence is Laya's head applied once, whatever the injection gate."""
    m, b = _tiny_zlaya()
    with torch.no_grad():
        m.inject.fill_(0.7)
        h = m.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"].long()).last_hidden_state
        h = h + m.type_emb(b["qtype"])[:, None, :]
        u = m.head(h, src_key_padding_mask=~b["attention_mask"])
        ref = m.readout(u, b)
        assert torch.allclose(m(b, steps=1)["logits"], ref, atol=1e-5)
        assert not torch.allclose(m(b, steps=2)["logits"], ref, atol=1e-3)


def test_trace_matches_separate_runs():
    m, b = _tiny_zlaya()
    with torch.no_grad():
        m.inject.fill_(0.3)
        tr = m(b, steps=4, trace=True)["trace"]
        for s in (1, 2, 4):
            assert torch.allclose(tr[s - 1], m(b, steps=s)["logits"], atol=1e-5)


def test_untied_uses_distinct_heads():
    m, b = _tiny_zlaya(untied_steps=3)
    assert len(m.extra) == 2
    with torch.no_grad():
        a = m(b, steps=1)["logits"]  # steps is fixed by untied_steps
        for p in m.extra[1].parameters():
            p.add_(0.1)
        assert not torch.allclose(a, m(b, steps=1)["logits"], atol=1e-4)


@pytest.mark.parametrize("split", ["train", "test"])
def test_nl_labels_match_sentences(split):
    rng = random.Random(0)
    for _ in range(200):
        ex = make_nl_example(rng, rng.randint(1, 24), split=split)
        vals = _eval_nl(ex.state)
        for qid, q in ex.questions.items():
            if q["type"] == "noul":
                continue
            who = re.findall(r"[A-Z]\w+", q["instructions"])[-1]
            want = vals[who] if q["type"] == "choice" else vals[who] // 2
            assert ex.labels[qid] == want


def test_split_uses_only_its_templates():
    rng = random.Random(1)
    held = [re.compile(t.replace(".", "\\.").replace("{x}", "\\w+").replace("{y}", "\\w+").replace("{v}", "\\d"))
            for t in templates("ROOT", "test") + templates("COPY", "test")]
    for split in ("train", "test"):
        ex = make_nl_example(rng, 6, split=split)
        for s in re.findall(r"[^.]+\.", ex.state):
            assert any(h.fullmatch(s.strip()) for h in held) == (split == "test")


def test_sequence_markers_point_at_mask_tokens():
    tok = pytest.importorskip("transformers").AutoTokenizer
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*"], local_files_only=True)
        tok = tok.from_pretrained(path + "/tokenizer")
    except Exception:
        pytest.skip("Laya tokenizer not in the local cache")
    ex = make_nl_example(random.Random(2), 5)
    seqs = [build_sequence(tok, ex.state, q) for q in ex.questions.values()]
    for (ids, markers), q in zip(seqs, ex.questions.values()):
        assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
        assert all(ids[i] == tok.mask_token_id for i in markers)
        assert len(markers) == {"choice": 10, "score": 5, "noul": 2}[q["type"]]
    b = collate(seqs, [QTYPES[q["type"]] for q in ex.questions.values()], tok.pad_token_id)
    assert b["marker_mask"].sum(-1).tolist() == [len(m) for _, m in seqs]
    add_facts(b, tok.sep_token_id, tok.convert_tokens_to_ids("."))
    n_facts = ex.state.count(".")
    assert (b["fact_mask"].any(-1).sum(-1) == n_facts).all()
    assert (b["head_len"] > torch.tensor([max(m) for _, m in seqs])).all()


SEP, DOT = 3, 4


def _fact_batch(rows):
    """rows: lists of sentence lengths; question part is [CLS] q q [SEP] [MASK] o [MASK] o [SEP]."""
    seqs = []
    for lens in rows:
        ids = [1, 7, 8, SEP, 2, 9, 2, 10, SEP]
        for n in lens:
            ids += list(range(20, 20 + n - 1)) + [DOT]
        seqs.append((ids + [SEP], [4, 6]))
    return add_facts(collate(seqs, [0] * len(seqs), 0), SEP, DOT)


def test_fact_spans():
    ids = [1, 7, SEP, 2, 9, SEP, 20, 21, DOT, 22, DOT, 23, SEP]
    assert fact_spans(ids, SEP, DOT) == (6, [(6, 9), (9, 11), (11, 12)])


def test_facts_padding_does_not_change_answers():
    """A sequence gives the same logits alone and batched with a longer one."""
    m, _ = _tiny_zlaya(units="facts")
    with torch.no_grad():
        m.inject.fill_(0.5)
        alone = _fact_batch([[3, 4]])
        both = _fact_batch([[3, 4], [5, 2, 6, 3]])
        ids = torch.randint(30, 128, both["input_ids"].shape)
        alone["input_ids"][:] = ids[:1, : alone["input_ids"].shape[1]]
        both["input_ids"] = torch.where(both["attention_mask"], ids, 0)
        a = m(alone, steps=3)["logits"]
        b = m(both, steps=3)["logits"]
        assert torch.allclose(a[0], b[0], atol=1e-4)


def test_facts_steps1_is_one_head_pass():
    m, _ = _tiny_zlaya(units="facts")
    b = _fact_batch([[3, 4, 2], [5, 2]])
    with torch.no_grad():
        m.inject.fill_(0.7)
        h = m.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"].long()).last_hidden_state
        e, pad = m.pool_facts(h, b)
        e = e + m.type_emb(b["qtype"])[:, None, :]
        assert e.shape[1] == 9 + 3 and pad[1, -1]  # 9 question tokens + 3 fact slots, row 1 has 2 facts
        ref = m.readout(m.head(e, src_key_padding_mask=pad), b)
        assert torch.allclose(m(b, steps=1)["logits"], ref, atol=1e-5)
