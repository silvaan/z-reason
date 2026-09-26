"""ZReasoner: a Jev-style typed-output model that reasons by iterating in latent space.

    f_i = Enc(fact_i)                           # each fact of the state -> one vector
    q   = Enc(question)                         # each question -> one vector
    e   = [q, f_1, ..., f_F]                    # one independent set per question
    z_0 ~ N(0, sigma^2 I)                       # like x_T in DDPM
    z_t = Core(z_{t-1} + e)                     # t = 1..steps, shared weights (input injection)
    y   = Coda(z_T)[question slot]
    p   = softmax(<W y, Enc(option_k)>)         # choice / score, answer set defined per request
    p   = sigmoid(w . y)                        # noul

Reasoning happens over fact vectors, not characters: one attention step can move
information from one fact to another (e.g. "B=a" reads the value of "a=7"), so
each latent step can perform one hop. `steps` is chosen at inference time;
training samples it at random so the same weights keep working for any number
of iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD


@dataclass
class ZConfig:
    vocab_size: int = 82
    d: int = 256
    heads: int = 8
    n_enc: int = 2       # fact / question / option encoder (character level)
    n_core: int = 2      # recurrent block, applied `steps` times
    n_coda: int = 1
    max_pos: int = 64    # max characters per fact
    mlp_mult: int = 4
    init_noise: float = 1.0
    untied_steps: int = 0  # >0: baseline with distinct core weights per step (fixed depth)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, cfg: ZConfig):
        super().__init__()
        d = cfg.d
        self.h = cfg.heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_mult * d, bias=False), nn.GELU(),
                                 nn.Linear(cfg.mlp_mult * d, d, bias=False))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.n2(x))


def pad_mask(valid: torch.Tensor) -> torch.Tensor:
    """valid (N, L) -> (N, 1, L, L) bool; padded rows attend to themselves (avoids NaN)."""
    eye = torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)
    return ((valid.unsqueeze(2) & valid.unsqueeze(1)) | eye).unsqueeze(1)


class ZReasoner(nn.Module):
    def __init__(self, cfg: ZConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_pos, d)
        self.enc = nn.ModuleList([Block(cfg) for _ in range(cfg.n_enc)])
        self.enc_norm = RMSNorm(d)
        self.kind = nn.Embedding(2, d)  # 0 = question slot, 1 = fact
        n_cores = max(cfg.untied_steps, 1)
        self.core = nn.ModuleList([nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)]) for _ in range(n_cores)])
        self.z_norm = RMSNorm(d)
        self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
        self.out_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.opt_proj = nn.Linear(d, d, bias=False)
        self.noul = nn.Linear(d, 1)

    # ------------------------------------------------------------------ parts
    def embed_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """(..., T) character ids (a CLS/BOS token at 0) -> (..., d) mean-pooled vector.
        Mean pooling: a vector read only at CLS starts out nearly identical for every
        text (cosine ~0.98 at init) and the model then fails to learn."""
        shape = tokens.shape[:-1]
        t = tokens.reshape(-1, tokens.shape[-1])
        valid = t != PAD
        valid[:, 0] = True
        pos = torch.arange(t.shape[1], device=t.device).clamp(max=self.cfg.max_pos - 1)
        x = self.tok(t) + self.pos(pos)
        mask = pad_mask(valid)
        for blk in self.enc:
            x = blk(x, mask)
        w = valid.unsqueeze(-1).to(x.dtype)
        return self.enc_norm((x * w).sum(1) / w.sum(1)).view(*shape, -1)

    def core_step(self, z, e, mask, i):
        u = z + e  # input injection: the encoded facts/question are re-read every step
        for blk in self.core[min(i, len(self.core) - 1)]:
            u = blk(u, mask)
        return self.z_norm(u)

    def readout(self, z, mask, batch, opt_emb):
        h = z
        for blk in self.coda:
            h = blk(h, mask)
        y = self.out_norm(h[:, 0])  # question slot
        logits = torch.einsum("nd,nkd->nk", self.q_proj(y), opt_emb) / math.sqrt(self.cfg.d)
        logits = logits.masked_fill(~batch["opt_valid"], float("-inf"))
        return {"logits": logits, "noul_logit": self.noul(y).squeeze(-1), "y": y}

    # ---------------------------------------------------------------- forward
    def forward(self, batch, steps: int, grad_steps: int | None = None, noise: float | None = None,
                trace: bool = False, generator: torch.Generator | None = None):
        """Run `steps` latent reasoning iterations.

        grad_steps: backprop only through the last `grad_steps` iterations (truncated BPTT).
        trace: also return readouts after every step (for anytime/steps analysis).
        """
        if self.cfg.untied_steps:
            steps = self.cfg.untied_steps
        qb = batch["q_batch"]
        facts = self.embed_text(batch["fact_tokens"])                  # (B, F, d)
        qv = self.embed_text(batch["q_tokens"])                        # (Nq, d)
        opt_emb = self.opt_proj(self.embed_text(batch["opt_tokens"]))  # (Nq, K, d)
        e = torch.cat([qv.unsqueeze(1), facts[qb]], 1)                 # (Nq, 1+F, d)
        kind = torch.ones(e.shape[1], dtype=torch.long, device=e.device)
        kind[0] = 0
        e = e + self.kind(kind)
        valid = torch.cat([torch.ones_like(qb, dtype=torch.bool).unsqueeze(1), batch["fact_valid"][qb]], 1)
        mask = pad_mask(valid)

        sigma = self.cfg.init_noise if noise is None else noise
        z = sigma * torch.randn(e.shape, device=e.device, dtype=e.dtype, generator=generator)
        n_nograd = 0 if grad_steps is None else max(steps - grad_steps, 0)
        traces = []
        for i in range(steps):
            if i < n_nograd:
                with torch.no_grad():
                    z = self.core_step(z, e, mask, i)
            else:
                z = self.core_step(z, e, mask, i)
            if trace:
                traces.append(self.readout(z, mask, batch, opt_emb))
        out = traces[-1] if trace else self.readout(z, mask, batch, opt_emb)
        if trace:
            out = dict(out, trace=traces)
        return out


# ---------------------------------------------------------------------- loss
def question_loss(out, batch):
    """Proper scoring rules (CE / BCE) => optimum is calibrated probabilities.
    Score questions additionally get an ordinal EMD term."""
    t, lab = batch["q_type"], batch["q_label"]
    logits = out["logits"].float()
    loss = torch.zeros((), device=logits.device)
    stats = {}
    cat = t <= 1
    if cat.any():
        ce = F.cross_entropy(logits[cat], lab[cat], reduction="none")
        loss = loss + ce.sum()
        stats["ce"] = ce.mean().item()
    sc = t == 1
    if sc.any():
        p = logits[sc].softmax(-1)
        onehot = F.one_hot(lab[sc], p.shape[-1]).float()
        emd = (p.cumsum(-1) - onehot.cumsum(-1)).pow(2).sum(-1)
        loss = loss + emd.sum()
    nl = t == 2
    if nl.any():
        bce = F.binary_cross_entropy_with_logits(out["noul_logit"][nl].float(), lab[nl].float(), reduction="none")
        loss = loss + bce.sum()
        stats["bce"] = bce.mean().item()
    return loss / len(t), stats


def correct(out, batch):
    t, lab = batch["q_type"], batch["q_label"]
    pred_cat = out["logits"].argmax(-1)
    pred_noul = (out["noul_logit"] > 0).long()
    pred = torch.where(t == 2, pred_noul, pred_cat)
    return pred == lab
