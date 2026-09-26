"""Turn Jev-style requests (state + typed questions) into padded tensors.

The state is split into *facts* (segments separated by ';' or newlines). Each
fact, each question and each answer option is a short character sequence that
the model encodes into a single vector. Latent reasoning then happens over the
set {question, fact_1, ..., fact_F}, one independent set per question.
"""

from __future__ import annotations

import json
import re

import torch

from .tokenizer import BOS, PAD, QRY, CharTokenizer

TYPE_ID = {"choice": 0, "score": 1, "noul": 2}
MAX_SEG = 32
_SPLIT = re.compile(r"[;\n]+")


def option_texts(q: dict) -> list[str]:
    crit = q.get("criteria")
    if q["type"] == "noul" or crit is None:
        return []
    if isinstance(crit, dict):
        return [f"{k}: {v}" if k != v else str(k) for k, v in crit.items()]
    return [str(c) for c in crit]


def option_names(q: dict) -> list[str]:
    crit = q["criteria"]
    return list(crit.keys()) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]


def instruction_text(q: dict) -> str:
    ins = q["instructions"]
    return ins if isinstance(ins, str) else json.dumps(ins)


def state_facts(state) -> list[str]:
    text = state if isinstance(state, str) else json.dumps(state)
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def _pad3(seqs: list[list[list[int]]], first: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Nested [group][item][token] -> (G, I, T) tokens and (G, I) validity. `first` is prepended (CLS)."""
    pre = [first] if first is not None else []
    G = len(seqs)
    I = max([len(s) for s in seqs] + [1])
    T = max([len(t) + len(pre) for s in seqs for t in s] + [1])
    out = torch.full((G, I, T), PAD, dtype=torch.long)
    valid = torch.zeros((G, I), dtype=torch.bool)
    for g, s in enumerate(seqs):
        for i, t in enumerate(s):
            t = pre + t
            out[g, i, : len(t)] = torch.tensor(t)
            valid[g, i] = True
    return out, valid


def collate(requests: list[dict], tok: CharTokenizer, labels: list[dict] | None = None) -> dict:
    """requests: [{"state": ..., "questions": {qid: spec}}]; labels: [{qid: int}] (optional)."""
    facts, qtok, q_b, q_type, q_label, q_opts, q_ids = [], [], [], [], [], [], []
    n_tokens = 0
    for b, req in enumerate(requests):
        f = [tok.encode(s)[:MAX_SEG] for s in state_facts(req["state"])] or [[]]
        facts.append(f)
        n_tokens += sum(map(len, f))
        for qid, q in req["questions"].items():
            q_b.append(b)
            q_ids.append((b, qid))
            q_type.append(TYPE_ID[q["type"]])
            qtok.append([tok.encode(instruction_text(q))[:MAX_SEG]])
            q_opts.append([tok.encode(t)[:MAX_SEG] for t in option_texts(q)])
            n_tokens += len(qtok[-1][0])
            if labels is not None:
                q_label.append(int(labels[b][qid]))

    fact_tokens, fact_valid = _pad3(facts, BOS)
    q_tokens, _ = _pad3(qtok, QRY)
    opt_tokens, opt_valid = _pad3(q_opts, BOS)
    out = {
        "fact_tokens": fact_tokens,          # (B, F, Ls)
        "fact_valid": fact_valid,            # (B, F)
        "q_tokens": q_tokens[:, 0],          # (Nq, Lq)
        "q_batch": torch.tensor(q_b, dtype=torch.long),
        "q_type": torch.tensor(q_type, dtype=torch.long),
        "opt_tokens": opt_tokens,            # (Nq, K, Lo)
        "opt_valid": opt_valid,              # (Nq, K)
        "q_ids": q_ids,
        "n_tokens": n_tokens,
    }
    if labels is not None:
        out["q_label"] = torch.tensor(q_label, dtype=torch.long)
    return out


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
