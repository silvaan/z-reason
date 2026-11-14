"""Evaluate a z-Laya run: accuracy and calibration as a function of (depth, steps).

    python eval_laya.py runs/zl_s0 [--depths 1-24] [--split train|test]
    python eval_laya.py runs/laya_zeroshot --zero-shot          # Laya as released

Writes runs/<name>/eval_<split>.json with the same layout as eval.py
(acc[step][depth], acc_by_type, calibration). `split` selects the sentence templates:
"train" uses the templates seen in training, "test" only the held-out ones. Encoder
states of the evaluation set are cached in cache/ and shared by all runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval import ece
from train_laya import encode_eval
from zreason.laya_head import add_facts, load_zlaya
from zreason.nl_tasks import make_nl_dataset

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
TYPES = {0: "choice", 1: "score", 2: "noul"}


def parse_range(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--depths", default="1-24")
    ap.add_argument("--n", type=int, default=60, help="states per depth (2 questions each)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    cfg = {} if args.zero_shot else json.loads((run / "config.json").read_text())["train"]
    untied = cfg.get("untied", 0)
    units = cfg.get("units", "tokens")  # Laya as released, and runs from before the option existed
    model, tok = load_zlaya(untied_steps=untied, units=units)
    model.encoder.to(torch.bfloat16)
    if not args.zero_shot:
        missing, unexpected = model.load_state_dict(torch.load(run / "head.pt"), strict=False)
        assert not unexpected and all(k.startswith("encoder.") for k in missing)
    model.eval()

    depths = parse_range(args.depths)
    cache = Path("cache") / f"nl_eval_{args.split}_{args.depths}_n{args.n}_s{args.seed}.pt"
    if cache.exists():
        batches = torch.load(cache)
    else:
        exs = make_nl_dataset(args.seed, args.n, depths, split=args.split)
        batches = encode_eval(model, tok, exs)
        cache.parent.mkdir(exist_ok=True)
        torch.save(batches, cache)

    steps = [untied] if untied else STEPS
    rows = []  # (depth, type, step, hit, conf)
    with torch.no_grad():
        for b in batches:
            if units == "facts":
                add_facts(b, model.sep_id, model.period_id)
            g = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(g, steps=max(steps), trace=True, h=g["h"].to(torch.bfloat16))
            for s in steps:
                lg = out["logits"] if untied else out["trace"][s - 1]
                p = lg.float().softmax(-1)
                hit = (p.argmax(-1) == g["label"]).cpu().numpy()
                conf = p.max(-1).values.cpu().numpy()
                for d, t, h_, c in zip(b["depth"].tolist(), b["qtype"].tolist(), hit, conf):
                    rows.append((d, t, s, bool(h_), float(c)))

    acc, acc_t, cal = defaultdict(dict), defaultdict(lambda: defaultdict(dict)), {}
    for s in steps:
        rs = [r for r in rows if r[2] == s]
        for d in sorted({r[0] for r in rs}):
            sel = [r[3] for r in rs if r[0] == d]
            acc[str(s)][str(d)] = float(np.mean(sel))
            for t, name in TYPES.items():
                sel = [r[3] for r in rs if r[0] == d and r[1] == t]
                if sel:
                    acc_t[str(s)][name][str(d)] = float(np.mean(sel))
        hit = np.array([r[3] for r in rs])
        conf = np.array([r[4] for r in rs])
        cal[str(s)] = {"ece": ece(conf, hit), "acc": float(hit.mean()), "mean_conf": float(conf.mean())}
    res = {"acc": acc, "acc_by_type": acc_t, "calibration": cal, "n": len(rows) // len(steps),
           "split": args.split, "train_depth": cfg.get("train_depth", 8), "zero_shot": args.zero_shot}
    (run / f"eval_{args.split}.json").write_text(json.dumps(res, indent=1))

    groups = [(1, 8), (9, 16), (17, 24)]
    print(f"{run.name} [{args.split}]  accuracy (all question types) by depth range")
    print("steps  " + "  ".join(f"{a:>2}-{b:<2}" for a, b in groups) + "   ECE")
    for s in steps:
        cells = []
        for a, b in groups:
            v = [acc[str(s)][str(d)] for d in range(a, b + 1) if str(d) in acc[str(s)]]
            cells.append(f"{100 * np.mean(v):5.1f}" if v else "    -")
        print(f"{s:>5}  " + "  ".join(cells) + f"  {cal[str(s)]['ece']:.3f}")


if __name__ == "__main__":
    main()
