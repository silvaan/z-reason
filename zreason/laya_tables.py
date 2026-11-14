"""Separately encoded sentences and questions for the NL variable tracking task.

The NL task draws from a closed set of strings: about 18k distinct fact sentences and 10k
distinct question blocks. Each one is encoded once by Laya's frozen encoder and kept in a
table on the GPU, so training batches are gathered by index and the data is unlimited, as in
v1. A fact is encoded alone ("[CLS] sentence [SEP]") and a question block in Laya's layout
without a state ("[CLS] <type> question: ... [SEP] [MASK] opt ... [SEP] [SEP]"), so the facts
of a state are shared by all its questions instead of being re-encoded with each one.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .laya_head import QTYPES, build_sequence
from .nl_tasks import (CHOICE_CRITERIA, CHOICE_Q, COPY, NOUL_Q, PEOPLE, ROOT, SCORE_CRITERIA, SCORE_Q,
                       Example)


def all_facts() -> list[str]:
    out = [t.format(x=x, v=v) for t in ROOT for x in PEOPLE for v in range(10)]
    out += [t.format(x=x, y=y) for t in COPY for x in PEOPLE for y in PEOPLE if x != y]
    return [f[0].upper() + f[1:] for f in out]


def all_questions() -> list[dict]:
    out = [{"type": "choice", "instructions": t.format(x=x), "criteria": CHOICE_CRITERIA} for t in CHOICE_Q for x in PEOPLE]
    out += [{"type": "score", "instructions": t.format(x=x), "criteria": SCORE_CRITERIA} for t in SCORE_Q for x in PEOPLE]
    out += [{"type": "noul", "instructions": t.format(x=x, y=y)} for t in NOUL_Q for x in PEOPLE for y in PEOPLE if x != y]
    return out


def split_facts(state: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.]+\.", state)]


@torch.no_grad()
def build_tables(model, tok, path: str | Path, bs: int = 256):
    """Encode every fact and question block once; save token states in fp16."""
    def encode(seqs):
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, model.encoder.config.hidden_size, dtype=torch.float16)
        for i in range(0, len(order), bs):
            idx = order[i: i + bs]
            n = max(len(seqs[j]) for j in idx)
            ids = torch.full((len(idx), n), tok.pad_token_id)
            att = torch.zeros((len(idx), n), dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, : len(seqs[j])] = torch.tensor(seqs[j])
                att[r, : len(seqs[j])] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encoder(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            out[idx, :n] = h.half().cpu()
        return out

    facts = all_facts()
    f_ids = [tok(f, add_special_tokens=False)["input_ids"] for f in facts]
    f_states = encode([[tok.cls_token_id] + s + [tok.sep_token_id] for s in f_ids])[:, 1:]  # drop [CLS]
    f_len = torch.tensor([len(s) for s in f_ids])                                      # and [SEP] by length

    qs = all_questions()
    q_seqs = [build_sequence(tok, "", q) for q in qs]
    q_states = encode([s for s, _ in q_seqs])
    K = max(len(m) for _, m in q_seqs)
    markers = torch.zeros(len(qs), K, dtype=torch.long)
    n_opt = torch.tensor([len(m) for _, m in q_seqs])
    for i, (_, m) in enumerate(q_seqs):
        markers[i, : len(m)] = torch.tensor(m)
    torch.save({"facts": facts, "f_states": f_states[:, : int(f_len.max())].contiguous(), "f_len": f_len,
                "questions": [(q["type"], q["instructions"]) for q in qs], "q_states": q_states,
                "q_len": torch.tensor([len(s) for s, _ in q_seqs]), "markers": markers, "n_opt": n_opt,
                "qtype": torch.tensor([QTYPES[q["type"]] for q in qs])}, path)


class Tables:
    def __init__(self, path: str | Path, device: str = "cuda"):
        t = torch.load(path)
        self.f_index = {f: i for i, f in enumerate(t["facts"])}
        self.q_index = {q: i for i, q in enumerate(t["questions"])}
        self.f_states, self.q_states = t["f_states"].to(device), t["q_states"].to(device)
        ar = lambda n: torch.arange(n, device=device)
        self.f_mask = ar(self.f_states.shape[1])[None] < t["f_len"].to(device)[:, None]
        self.q_valid = ar(self.q_states.shape[1])[None] < t["q_len"].to(device)[:, None]
        self.markers = t["markers"].to(device)
        self.m_mask = ar(self.markers.shape[1])[None] < t["n_opt"].to(device)[:, None]
        self.qtype = t["qtype"].to(device)
        self.device = device

    def fact_stats(self):
        """Per-dimension mean and std over all fact tokens (for ZLaya.fact_mu / fact_sd)."""
        x = self.f_states[self.f_mask].float()
        return x.mean(0), x.std(0) + 1e-5

    def batch(self, exs: list[Example]) -> dict:
        """One row per question; the facts of a state are repeated for each of its questions."""
        q_idx, f_rows, labels, depth = [], [], [], []
        for e in exs:
            fi = [self.f_index[f] for f in split_facts(e.state)]
            for qid, q in e.questions.items():
                q_idx.append(self.q_index[(q["type"], q["instructions"])])
                f_rows.append(fi)
                labels.append(e.labels[qid])
                depth.append(e.meta[qid]["depth"])
        F = max(len(r) for r in f_rows)
        fidx = torch.full((len(f_rows), F), -1, dtype=torch.long)
        for i, r in enumerate(f_rows):
            fidx[i, : len(r)] = torch.tensor(r)
        fidx = fidx.to(self.device)
        qi = torch.tensor(q_idx, device=self.device)
        has = fidx >= 0
        return {"q_h": self.q_states[qi], "q_valid": self.q_valid[qi],
                "marker_pos": self.markers[qi], "marker_mask": self.m_mask[qi], "qtype": self.qtype[qi],
                "fact_tok": self.f_states[fidx.clamp(min=0)],
                "fact_mask": self.f_mask[fidx.clamp(min=0)] & has[:, :, None],
                "label": torch.tensor(labels, device=self.device), "depth": torch.tensor(depth, device=self.device)}
