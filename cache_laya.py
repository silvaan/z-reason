"""Precompute Laya's frozen encoder states for a fixed training set of the NL task.

The encoder is frozen in every z-Laya run, so its output is computed once and shared by all
runs. Each sequence is one question (Laya's layout); its states are stored in fp16 in one flat
memmap, with offsets and metadata in index.pt.

Example:
    python cache_laya.py --out cache/nl_train --n-states 30000 --seed 0
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from zreason.laya_head import QTYPES, build_sequence, collate, load_zlaya
from zreason.nl_tasks import make_nl_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=30000)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--n-questions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=48)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_zlaya()
    model.encoder.to(torch.bfloat16).eval()

    rng = random.Random(args.seed)
    seqs, meta = [], []
    for _ in range(args.n_states):
        e = make_nl_example(rng, rng.randint(1, args.max_depth), n_questions=args.n_questions)
        for qid, q in e.questions.items():
            seqs.append(build_sequence(tok, e.state, q))
            meta.append((QTYPES[q["type"]], e.labels[qid], e.meta[qid]["depth"]))
    lens = np.array([len(s) for s, _ in seqs])
    offsets = np.concatenate([[0], np.cumsum(lens)])
    d = model.encoder.config.hidden_size
    print(f"{len(seqs)} sequences, {offsets[-1]} tokens, {offsets[-1] * d * 2 / 1e9:.1f} GB", flush=True)

    states = np.lib.format.open_memmap(out / "states.npy", mode="w+", dtype=np.float16, shape=(int(offsets[-1]), d))
    order = np.argsort(lens)  # length-sorted batches waste little on padding
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(order), args.bs):
            idx = order[i: i + args.bs]
            b = collate([seqs[j] for j in idx], [0] * len(idx), tok.pad_token_id)
            h = model.encode({k: v.cuda() for k, v in b.items()}).half().cpu().numpy()
            for r, j in enumerate(idx):
                states[offsets[j]: offsets[j + 1]] = h[r, : lens[j]]
            if i // args.bs % 100 == 0:
                print(f"{i + len(idx)}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    states.flush()
    torch.save({"ids": [s for s, _ in seqs], "markers": [m for _, m in seqs], "offsets": offsets,
                "qtype": [m[0] for m in meta], "label": [m[1] for m in meta], "depth": [m[2] for m in meta],
                "args": vars(args)}, out / "index.pt")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
