"""Anytime behaviour on single requests: P(correct answer) after each latent step,
for choice questions about variables at different depths, averaged over programs.

    python -m analysis.trace_figure runs/zr_s0 --out figures/fig0_trace.png
"""

from __future__ import annotations

import argparse
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plots import GRID, INK2, RAMP
from zreason.api import ZReason
from zreason.batching import collate, to_device
from zreason.tasks import CHOICE_CRITERIA, _build_program


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="figures/fig0_trace.png")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    api = ZReason.load(args.run)
    rng = random.Random(3)
    depths = [1, 2, 4, 8, 12, 16]
    probs = {d: [] for d in depths}
    for _ in range(args.n):
        names, parent, dep, value, state = _build_program(rng, 16, rng.randint(24, 40))
        chain = names[:17]  # the main chain: chain[d] has depth d
        qs = {f"d{d}": {"type": "choice", "instructions": f"val {chain[d]}", "criteria": CHOICE_CRITERIA} for d in depths}
        b = to_device(collate([{"state": state, "questions": qs}], api.tok), api.device)
        out = api.model(b, steps=args.steps, trace=True, generator=torch.Generator(device=api.device).manual_seed(0))
        for i, (_, qid) in enumerate(b["q_ids"]):
            d = int(qid[1:])
            target = value[chain[d]]
            probs[d].append([o["logits"][i].float().softmax(-1)[target].item() for o in out["trace"]])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.arange(1, args.steps + 1)
    for c, d in zip(RAMP, depths):
        p = np.array(probs[d])
        ax.plot(xs, p.mean(0), color=c, marker="o", ms=3.5, label=f"depth {d}" + (", unseen" if d > 8 else ""),
                ls="-" if d <= 8 else (0, (4, 2)))
    ax.axhline(0.1, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(1.2, 0.12, "uniform 0.1", ha="left", fontsize=8, color=INK2)
    ax.set_xlabel("latent step t")
    ax.set_ylabel("P(correct value)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs[::1] if args.steps <= 16 else xs[::2])
    ax.set_title("The answer sharpens step by step; deeper chains take longer")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
