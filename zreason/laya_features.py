"""Frozen Laya sentence vectors as the input of the v1 model.

The v1 model encodes every fact, question and answer option with a small character-level
encoder trained from scratch. Here that encoder is replaced by Laya's frozen ModernBERT-large:
each distinct string the NL task can produce is encoded once, alone ("[CLS] text [SEP]"), its
tokens are mean pooled, and the vectors are standardized per dimension with statistics of the
fact vectors (the raw pooled vectors of one state have mean cosine 0.92). The model then maps
them to its width with a trainable adapter (ZConfig.ext_dim), and the recurrence is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .batching import TYPE_ID, instruction_text, option_texts, state_facts
from .laya_tables import all_facts, all_questions


def nl_strings() -> tuple[list[str], set[str]]:
    """Every text of the NL task, and the fact sentences whose statistics standardize them."""
    qs = all_questions()
    facts = set(all_facts())
    return sorted(facts | {instruction_text(q) for q in qs} | {o for q in qs for o in option_texts(q)}), facts


@torch.no_grad()
def build_vectors(path: str | Path, strings: list[str] | None = None, stat_strings: set[str] | None = None,
                  bs: int = 256):
    """Encode `strings` (default: the NL task) and standardize with the statistics of `stat_strings`."""
    from .laya_head import load_zlaya

    if strings is None:
        strings, stat_strings = nl_strings()
    model, tok = load_zlaya(units="tokens")
    model.encoder.to(torch.bfloat16).eval()
    ids = [tok(s, add_special_tokens=False)["input_ids"] for s in strings]
    vecs = torch.zeros(len(strings), model.encoder.config.hidden_size)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for i in range(0, len(order), bs):
        idx = order[i: i + bs]
        n = max(len(ids[j]) for j in idx) + 2
        x = torch.full((len(idx), n), tok.pad_token_id)
        att = torch.zeros((len(idx), n), dtype=torch.long)
        for r, j in enumerate(idx):
            s = [tok.cls_token_id] + ids[j] + [tok.sep_token_id]
            x[r, : len(s)] = torch.tensor(s)
            att[r, 1: len(s) - 1] = 1  # pool over the text tokens only
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encoder(input_ids=x.cuda(), attention_mask=(x != tok.pad_token_id).long().cuda()).last_hidden_state
        w = att.cuda().unsqueeze(-1).float()
        vecs[idx] = ((h.float() * w).sum(1) / w.sum(1)).cpu()
    facts = torch.tensor([s in stat_strings for s in strings])
    mu, sd = vecs[facts].mean(0), vecs[facts].std(0) + 1e-5
    torch.save({"strings": strings, "vecs": (vecs - mu) / sd}, path)


class VectorTable:
    def __init__(self, path: str | Path):
        t = torch.load(path)
        self.index = {s: i for i, s in enumerate(t["strings"])}
        self.vecs = t["vecs"]

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.vecs[[self.index[s] for s in texts]] if texts else self.vecs.new_zeros(0, self.vecs.shape[1])


def _pad(groups: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    n = max([len(g) for g in groups] + [1])
    out = torch.zeros(len(groups), n, groups[0].shape[-1])
    valid = torch.zeros(len(groups), n, dtype=torch.bool)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
        valid[i, : len(g)] = True
    return out, valid


def collate_vec(requests: list[dict], table: VectorTable, labels: list[dict] | None = None) -> dict:
    """Same batch layout as batching.collate, with vectors (*_vec) in place of character tokens."""
    facts, q_vec, opts, q_b, q_type, q_label, q_ids = [], [], [], [], [], [], []
    for b, req in enumerate(requests):
        facts.append(table(state_facts(req["state"])))
        for qid, q in req["questions"].items():
            q_b.append(b)
            q_ids.append((b, qid))
            q_type.append(TYPE_ID[q["type"]])
            q_vec.append(table([instruction_text(q)])[0])
            opts.append(table(option_texts(q)))
            if labels is not None:
                q_label.append(int(labels[b][qid]))
    fact_vec, fact_valid = _pad(facts)
    opt_vec, opt_valid = _pad(opts)
    out = {"fact_vec": fact_vec, "fact_valid": fact_valid, "q_vec": torch.stack(q_vec),
           "q_batch": torch.tensor(q_b, dtype=torch.long), "q_type": torch.tensor(q_type, dtype=torch.long),
           "opt_vec": opt_vec, "opt_valid": opt_valid, "q_ids": q_ids, "n_tokens": 0}
    if labels is not None:
        out["q_label"] = torch.tensor(q_label, dtype=torch.long)
    return out
