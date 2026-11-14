"""z-reason on top of an open System-One model (Laya).

Laya (convaiinnovations/laya, Apache-2.0) is a one-pass typed decision model:

    h = Enc([CLS] <type> question: ins [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP] state [SEP])
    h = h + type_emb(type)
    u = Head(h)                                   # 2 pre-norm transformer layers
    logits_k = Scorer(u[mask_k])                  # one logit per option marker

ZLaya keeps the encoder, the scorer and the head weights and turns the head into the
weight-tied block of z-reason:

    u_1     = Head(h)                             # steps=1 is exactly Laya
    u_{t+1} = Head(u_t + g * h)                   # g is learned and starts at 0

so the one-pass model is the first step of the recurrence and more steps can only
add computation on top of it. The encoder is frozen; only the head, scorer, type
embedding and injection gate are trained.

With `units="tokens"` the recurrence runs over every token, as above. Like the
character-level recurrence of v1, this learns direct lookups but not their composition.
With `units="facts"` (the default) the state tokens are replaced by one vector per
sentence before the recurrence: a trainable layer attends within each sentence and its
output is mean pooled. The question and option tokens stay as they are, so the scorer
still reads Laya's option markers:

    e       = [h_question_and_options ; FactPool(h_sentence_1) ; ... ]
    u_1     = Head(e)
    u_{t+1} = Head(u_t + g * e)

Here steps=1 is Laya's head over pooled facts rather than Laya itself, and the one-pass
baseline is the same architecture trained with steps=1.

The sequence format below reproduces Laya's `build_sequence` (laya 0.4.18,
`laya/common.py`) for string states and the default sequential option layout. We
reimplement it instead of importing the package so the weights load with nothing
but transformers and safetensors.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LAYA_ID = "convaiinnovations/laya"
QTYPES = {"choice": 0, "score": 1, "noul": 2}


# ------------------------------------------------------------------ sequences
def render_options(q: dict) -> list[str]:
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        return [str(k) if v is None or v == "" else f"{k}: {v}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    return ["false: no, the statement does not hold", "true: yes, the statement holds"]


def build_sequence(tok, state: str, q: dict, max_len: int = 512, head_max_len: int = 192):
    """Token ids and option marker positions, in Laya's layout. Raises if the state is truncated,
    so a task never silently drops facts."""
    enc = lambda s, **kw: tok(s, add_special_tokens=False, **kw)["input_ids"]
    head_ids = enc(f"{q['type']} question: {q['instructions'].replace(tok.mask_token, ' ')}")
    opt_ids = [[tok.mask_token_id] + enc(" " + o.replace(tok.mask_token, " "), truncation=True, max_length=48)
               for o in render_options(q)]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    ids = [tok.cls_token_id] + head_ids[: max(8, budget)] + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    st = enc(state.replace(tok.mask_token, " "))
    room = max(0, max_len - len(ids) - 1)
    if len(st) > room:
        raise ValueError(f"state has {len(st)} tokens but only {room} fit")
    return ids + st + [tok.sep_token_id], markers


def fact_spans(ids: list[int], sep_id: int, period_id: int) -> tuple[int, list[tuple[int, int]]]:
    """Length of the question part (up to the [SEP] that closes the options) and the token span
    of every sentence of the state, which ends at a "." token."""
    head_len = [i for i, t in enumerate(ids) if t == sep_id][1] + 1
    spans, start = [], head_len
    for i in range(head_len, len(ids) - 1):
        if ids[i] == period_id:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(ids) - 1:
        spans.append((start, len(ids) - 1))
    return head_len, spans


def collate(seqs: list[tuple[list[int], list[int]]], qtypes: list[int], pad_id: int, labels=None):
    n, L = len(seqs), max(len(s) for s, _ in seqs)
    K = max(len(m) for _, m in seqs)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.bool)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    for i, (s, m) in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = True
        mpos[i, : len(m)] = torch.tensor(m)
        mmask[i, : len(m)] = True
    b = {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
         "qtype": torch.tensor(qtypes)}
    if labels is not None:
        b["label"] = torch.tensor(labels)
    return b


def add_facts(b: dict, sep_id: int, period_id: int) -> dict:
    """Add the sentence layout used by `units="facts"`: the question length and, for every
    sentence, the positions of its tokens."""
    spans = [fact_spans(r[m].tolist(), sep_id, period_id) for r, m in zip(b["input_ids"], b["attention_mask"])]
    n = len(spans)
    F_ = max(len(sp) for _, sp in spans)
    T = max(e - s for _, sp in spans for s, e in sp)
    b["head_len"] = torch.tensor([hl for hl, _ in spans])
    b["fact_idx"] = torch.zeros((n, F_, T), dtype=torch.long)
    b["fact_mask"] = torch.zeros((n, F_, T), dtype=torch.bool)
    for i, (_, sp) in enumerate(spans):
        for f, (s, e) in enumerate(sp):
            b["fact_idx"][i, f, : e - s] = torch.arange(s, e)
            b["fact_mask"][i, f, : e - s] = True
    return b


# ---------------------------------------------------------------------- model
class ZLaya(nn.Module):
    def __init__(self, encoder: nn.Module, head_layers: int = 2, untied_steps: int = 0, units: str = "facts"):
        super().__init__()
        assert units in ("tokens", "facts")
        self.units = units
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = lambda: nn.TransformerEncoderLayer(d, nhead, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer(), head_layers, enable_nested_tensor=False)
        # baseline: distinct head weights per step (a deeper one-pass head, same compute)
        self.extra = nn.ModuleList(nn.TransformerEncoder(layer(), head_layers, enable_nested_tensor=False)
                                   for _ in range(max(untied_steps - 1, 0)))
        self.untied_steps = untied_steps
        self.checkpointing = False  # recompute each step in backward: memory flat in the number of steps
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.inject = nn.Parameter(torch.zeros(d))
        if units == "facts":
            self.fact_enc = layer()
            self.fact_emb = nn.Parameter(torch.zeros(d))
            # the head has no positions, so who copies whom would rest on the frozen encoder alone
            self.fact_pos = nn.Embedding(64, d)
            nn.init.zeros_(self.fact_pos.weight)
            # Frozen-encoder states share one dominant direction: the pooled facts of a state have
            # mean cosine 0.92, and nothing is learned. Standardizing tokens with dataset
            # statistics (fit_fact_norm) brings it to 0.13.
            self.register_buffer("fact_mu", torch.zeros(d))
            self.register_buffer("fact_sd", torch.ones(d))
        self.register_buffer("temperature", torch.ones(3))

    @torch.no_grad()
    def encode(self, b):
        return self.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"].long()).last_hidden_state

    def pool_facts(self, h, b):
        """[question and option tokens ; one vector per sentence] and its padding mask.

        Either from one joint encoding `h` of question and state (Laya's layout, sentence spans
        from `add_facts`), or from separately encoded sentences (`laya_tables`): b["q_h"] holds
        the question block and b["fact_tok"] the tokens of every sentence."""
        if "fact_tok" in b:
            tok_all, head, head_valid = b["fact_tok"], b["q_h"], b["q_valid"]
        else:
            H = int(b["head_len"].max())
            tok_all = torch.gather(h, 1, b["fact_idx"].flatten(1)[:, :, None].expand(-1, -1, h.size(-1)))
            tok_all = tok_all.view(*b["fact_idx"].shape, -1)
            head = h[:, :H]
            head_valid = torch.arange(H, device=h.device)[None] < b["head_len"][:, None]
        valid = b["fact_mask"].any(-1)                                              # (B, F)
        tok = tok_all[valid]                                                         # (N, T, d)
        m = b["fact_mask"][valid]
        tok = (tok - self.fact_mu.to(tok.dtype)) / self.fact_sd.to(tok.dtype)
        tok = tok + self.fact_pos(torch.arange(tok.shape[1], device=tok.device).clamp(max=63))
        tok = self.fact_enc(tok, src_key_padding_mask=~m)
        w = m.unsqueeze(-1).to(tok.dtype)
        facts = head.new_zeros(*valid.shape, head.size(-1))
        facts[valid] = ((tok * w).sum(1) / w.sum(1)).to(head.dtype)
        facts = facts + self.fact_emb
        return torch.cat([head, facts], 1), ~torch.cat([head_valid, valid], 1)

    def head_step(self, u, pad, i):
        blk = self.head if i == 0 or not self.untied_steps else self.extra[min(i, self.untied_steps - 1) - 1]
        for layer in blk.layers:
            u = layer(u, src_key_padding_mask=pad)
        return u

    def readout(self, u, b):
        idx = b["marker_pos"][:, :, None].expand(-1, -1, u.size(-1))
        logits = self.scorer(torch.gather(u, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~b["marker_mask"], -1e4)

    def forward(self, b, steps: int = 1, grad_steps: int | None = None, trace: bool = False, h=None):
        """Logits after `steps` head iterations. `h` lets a caller reuse encoder states across
        step counts; `grad_steps` backpropagates only through the last iterations; `trace` also
        returns the logits after every step."""
        if self.untied_steps:
            steps = self.untied_steps
        if "fact_tok" in b:  # separately encoded sentences (laya_tables)
            h, pad = self.pool_facts(None, b)
        else:
            if h is None:
                h = self.encode(b)
            pad = ~b["attention_mask"]
            if self.units == "facts":
                h, pad = self.pool_facts(h, b)
        h = h + self.type_emb(b["qtype"])[:, None, :]
        n_nograd = 0 if grad_steps is None else max(steps - grad_steps, 0)
        u, traces = h, []
        for i in range(steps):
            x = u if i == 0 else u + self.inject * h
            if i < n_nograd:
                with torch.no_grad():
                    u = self.head_step(x, pad, i)
            elif self.checkpointing and self.training and torch.is_grad_enabled():
                u = checkpoint(self.head_step, x, pad, i, use_reentrant=False)
            else:
                u = self.head_step(x, pad, i)
            if trace:
                traces.append(self.readout(u, b))
        out = {"logits": traces[-1] if trace else self.readout(u, b)}
        if trace:
            out["trace"] = traces
        return out


def load_zlaya(path: str | None = None, untied_steps: int = 0, units: str = "facts", device: str = "cuda"):
    """ZLaya with Laya's English checkpoint (ModernBERT-large encoder) and its tokenizer."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    path = path or snapshot_download(LAYA_ID, allow_patterns=[
        "model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
    ecfg = AutoConfig.from_pretrained(os.path.join(path, "encoder"))
    rope = getattr(ecfg, "rope_parameters", None)  # transformers>=5 layout -> attributes 4.x reads
    if isinstance(rope, dict):
        for kind, attr in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
            if isinstance(rope.get(kind), dict) and hasattr(ecfg, attr):
                setattr(ecfg, attr, float(rope[kind]["rope_theta"]))
    model = ZLaya(AutoModel.from_config(ecfg, attn_implementation="sdpa"), untied_steps=untied_steps, units=units)
    w = {k: v for k, v in load_file(os.path.join(path, "model.safetensors")).items() if not k.startswith("act_head.")}
    missing, unexpected = model.load_state_dict(w, strict=False)
    assert not unexpected, unexpected
    assert all(k in ("inject", "fact_emb", "fact_mu", "fact_sd") or k.startswith(("extra.", "fact_enc.", "fact_pos.")) for k in missing), missing
    if units == "facts":  # the sentence layer starts as a copy of Laya's first head layer
        model.fact_enc.load_state_dict(model.head.layers[0].state_dict())
    for i, blk in enumerate(model.extra):  # untied baseline: every step starts from Laya's head
        blk.load_state_dict(model.head.state_dict())
    model.encoder.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    model.sep_id, model.period_id = tok.sep_token_id, tok.convert_tokens_to_ids(".")
    return model.to(device), tok


# ----------------------------------------------------------------------- loss
def question_loss(logits, b):
    """Cross entropy over the option markers (a proper scoring rule, as in v1), plus the ordinal
    EMD term for score questions."""
    loss = F.cross_entropy(logits, b["label"], reduction="none")
    sc = b["qtype"] == QTYPES["score"]
    if sc.any():
        p = logits[sc].softmax(-1)
        onehot = F.one_hot(b["label"][sc], p.shape[-1]).float()
        loss[sc] = loss[sc] + (p.cumsum(-1) - onehot.cumsum(-1)).pow(2).sum(-1)
    return loss.mean()
