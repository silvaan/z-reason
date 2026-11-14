"""ProofWriter: confusion between True, False and Unknown by proof depth.

Accuracy alone hides which answers fail. This counts (label, prediction) pairs on the whole
test set at the given steps; proof depths above 5 (a few Unknown questions whose depth is that
of a failed proof) are merged into 5.

    python -m analysis.pw_confusion runs/pw_s0 --steps 1 2 4 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval import _step_outputs
from zreason import proofwriter
from zreason.api import ZReason
from zreason.batching import collate, to_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pw", default="depth-5/test")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 16])
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    api = ZReason.load(args.run)
    model = api.model
    steps = [model.cfg.untied_steps] if model.cfg.untied_steps else args.steps
    exs = proofwriter.load(*args.pw.split("/"))
    conf = {s: np.zeros((6, 3, 3), dtype=int) for s in steps}  # step -> depth x label x prediction
    for i in range(0, len(exs), args.bs):
        chunk = exs[i: i + args.bs]
        b = to_device(collate([{"state": e.state, "questions": e.questions} for e in chunk], api.tok,
                              labels=[e.labels for e in chunk]), api.device)
        depth = [min(chunk[bi].meta[q]["depth"], 5) for bi, q in b["q_ids"]]
        lab = b["q_label"].tolist()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, steps=max(steps), trace=True, generator=torch.Generator(device=api.device).manual_seed(i))
        for s in steps:
            _, pred, _ = _step_outputs(out["trace"][s - 1], b["q_type"])
            for d, l, p in zip(depth, lab, pred.tolist()):
                conf[s][d, l, p] += 1
    res = {"labels": list(proofwriter.ANSWERS), "pw": args.pw,
           "confusion": {s: c.tolist() for s, c in conf.items()}}
    tag = args.pw.split("/")[0]
    (Path(args.run) / f"confusion_{tag}.json").write_text(json.dumps(res))
    for s, c in conf.items():
        print(f"steps {s}: depth -> [[TT TF TU] [FT FF FU] [UT UF UU]]")
        for d in range(6):
            print(f"  {d}: {c[d].tolist()}")


if __name__ == "__main__":
    main()
