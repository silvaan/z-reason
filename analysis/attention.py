"""Mechanistic check: which ancestor does each fact attend to at latent step t?

For every fact "x=parent" on a chain, we take the fact it attends to most strongly
(per core block and head) and record the chain distance from x to that fact:
k>0 means "the ancestor k hops up", 0 means itself, -1 means off-chain / not an ancestor.
One hop per step predicts k=1 at every step; pointer jumping predicts k ~ 2^(t-1).

    python -m analysis.attention runs/zr_s0 --depth 16 --n 200 --steps 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from zreason.api import ZReason
from zreason.batching import collate, state_facts, to_device
from zreason.model import pad_mask
from zreason.tasks import make_chain_example


def chain_distance(facts: list[str]):
    """-> dist[i][j] = hops from fact i up to fact j if j is an ancestor (0 = itself), else -1."""
    name = [f.split("=")[0] for f in facts]
    rhs = [f.split("=")[1] for f in facts]
    idx = {n: i for i, n in enumerate(name)}
    parent = [idx.get(r) if not r.isdigit() else None for r in rhs]
    F = len(facts)
    dist = np.full((F, F), -1, dtype=int)
    depth = np.zeros(F, dtype=int)
    for i in range(F):
        j, k = i, 0
        while j is not None:
            dist[i, j] = k
            j, k = parent[j], k + 1
        depth[i] = k - 1
    return dist, depth


@torch.no_grad()
def attention_maps(model, batch, steps, seed=0):
    """Yields (step, block, weights (Nq, H, L, L)) for the core blocks."""
    qb = batch["q_batch"]
    facts = model.embed_text(batch["fact_tokens"])
    qv = model.embed_text(batch["q_tokens"])
    e = torch.cat([qv.unsqueeze(1), facts[qb]], 1)
    kind = torch.ones(e.shape[1], dtype=torch.long, device=e.device)
    kind[0] = 0
    e = e + model.kind(kind)
    valid = torch.cat([torch.ones_like(qb, dtype=torch.bool).unsqueeze(1), batch["fact_valid"][qb]], 1)
    mask = pad_mask(valid)
    g = torch.Generator(device=e.device).manual_seed(seed)
    z = model.cfg.init_noise * torch.randn(e.shape, device=e.device, generator=g)
    for t in range(steps):
        u = z + e
        for b, blk in enumerate(model.core[0]):
            B, L, D = u.shape
            q, k, _ = blk.qkv(blk.n1(u)).view(B, L, 3, blk.h, D // blk.h).permute(2, 0, 3, 1, 4)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // blk.h)
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            yield t + 1, b, att
            u = blk(u, mask)
        z = model.z_norm(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    api = ZReason.load(args.run)
    rng = random.Random(args.seed)
    exs = [make_chain_example(rng, args.depth, n_questions=1) for _ in range(args.n)]
    # counts[(step, block, head)][k] over facts with enough ancestors
    counts = defaultdict(Counter)
    for i in range(0, len(exs), 50):
        chunk = exs[i: i + 50]
        b = to_device(collate([{"state": e.state, "questions": e.questions} for e in chunk], api.tok), api.device)
        dists = [chain_distance(state_facts(e.state)) for e in chunk]
        for t, blk, att in attention_maps(api.model, b, args.steps):
            top = att[:, :, 1:, 1:].argmax(-1).cpu().numpy()  # fact->fact, (Nq, H, F, F) -> (Nq, H, F)
            for n, (dist, depth) in enumerate(dists):
                F = len(depth)
                for h in range(top.shape[1]):
                    for f in range(F):
                        if depth[f] >= 2 ** (t - 1):  # enough ancestors for a 2^(t-1) jump
                            counts[(t, blk, h)][int(dist[f, top[n, h, f]])] += 1

    # per step: the head whose top-1 target is most often an ancestor; report its modal distance
    summary = {}
    print("step | best (block, head) | modal ancestor distance | share of facts | 2^(t-1)")
    for t in range(1, args.steps + 1):
        best = None
        for (tt, blk, h), c in counts.items():
            if tt != t:
                continue
            tot = sum(c.values())
            anc = {k: v for k, v in c.items() if k > 0}
            if not anc:
                continue
            k_mode, v = max(anc.items(), key=lambda kv: kv[1])
            if best is None or v / tot > best[3]:
                best = (blk, h, k_mode, v / tot, dict(sorted(c.items())))
        if best:
            blk, h, k_mode, share, dist = best
            summary[t] = {"block": blk, "head": h, "modal_distance": k_mode, "share": share, "hist": dist}
            print(f"{t:>4} | ({blk}, {h}) | {k_mode:>3} | {share:.2f} | {2 ** (t - 1)}")
    out = Path(args.run) / f"attention_depth{args.depth}.json"
    out.write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
