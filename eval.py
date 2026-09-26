"""Evaluate a trained run: accuracy / calibration as a function of (difficulty, steps).

    python eval.py runs/zr_s0 [--depths 1-16] [--max-steps 64]

Writes runs/<name>/eval.json. A single forward pass with trace=True gives the
readout after every step, so the whole steps axis costs one run of max_steps.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from zreason.api import ZReason
from zreason.batching import collate, to_device
from zreason.tasks import make_dataset

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(e)


def reliability(conf, hit, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()),
             "conf": float(conf[idx == b].mean()) if (idx == b).any() else None,
             "acc": float(hit[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]


def _step_outputs(o, qtype):
    """-> probability vector used for halting, prediction, confidence (all per question)."""
    p = o["logits"].float().softmax(-1).nan_to_num(0.0)
    pn = torch.sigmoid(o["noul_logit"].float())
    is_noul = qtype == 2
    pred = torch.where(is_noul, (pn > 0.5).long(), p.argmax(-1))
    conf = torch.where(is_noul, torch.maximum(pn, 1 - pn), p.max(-1).values)
    return torch.cat([p, pn.unsqueeze(-1)], -1), pred, conf


@torch.no_grad()
def run_eval(api: ZReason, exs, max_steps: int, bs: int = 128, seed: int = 0, tol: float = 1e-3):
    """Returns rows (depth, type, step, hit, conf) for the fixed steps grid and
    auto_rows (depth, type, halting step, hit) for steps="auto" (stop once no answer
    probability moves by more than `tol` between consecutive steps)."""
    model, dev = api.model, api.device
    steps = [s for s in STEPS if s <= max_steps]
    if model.cfg.untied_steps:
        steps = [model.cfg.untied_steps]
    rows, auto_rows = [], []
    for i in range(0, len(exs), bs):
        chunk = exs[i: i + bs]
        b = to_device(collate([{"state": e.state, "questions": e.questions} for e in chunk], api.tok,
                              labels=[e.labels for e in chunk]), dev)
        depth = np.array([chunk[bi].meta[qid]["depth"] for bi, qid in b["q_ids"]])
        qtype = b["q_type"].cpu().numpy()
        lab = b["q_label"]
        g = torch.Generator(device=dev).manual_seed(seed + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, steps=max(steps), trace=True, generator=g)
        per_step = [_step_outputs(o, b["q_type"]) for o in out["trace"]]
        if model.cfg.untied_steps:
            per_step = [None] * (len(per_step) - 1) + [per_step[-1]]
        for s in steps:
            _, pred, conf = per_step[s - 1]
            hit = (pred == lab).cpu().numpy()
            conf = conf.cpu().numpy()
            for j in range(len(depth)):
                rows.append((int(depth[j]), int(qtype[j]), s, bool(hit[j]), float(conf[j])))
        if not model.cfg.untied_steps:
            P = torch.stack([ps[0] for ps in per_step])                      # (S, Nq, K+1)
            moved = (P[1:] - P[:-1]).abs().amax(-1) >= tol                   # (S-1, Nq)
            still = torch.cat([torch.ones_like(moved[:1]), moved], 0)        # step 1 always "moving"
            # halting step = first step t (1-indexed, t>=2) whose update moved nothing
            idx = torch.arange(1, P.shape[0] + 1, device=P.device).unsqueeze(1).expand_as(still)
            halt = torch.where(~still, idx, torch.full_like(idx, P.shape[0])).amin(0)
            preds = torch.stack([ps[1] for ps in per_step])                  # (S, Nq)
            hit_auto = (preds.gather(0, (halt - 1).unsqueeze(0))[0] == lab).cpu().numpy()
            halt = halt.cpu().numpy()
            for j in range(len(depth)):
                auto_rows.append((int(depth[j]), int(qtype[j]), int(halt[j]), bool(hit_auto[j])))
    return rows, auto_rows


def summarize_auto(auto_rows):
    by = defaultdict(list)
    for d, t, s, h in auto_rows:
        by[d].append((s, h))
    return {d: {"mean_steps": float(np.mean([s for s, _ in v])), "median_steps": float(np.median([s for s, _ in v])),
                "acc": float(np.mean([h for _, h in v]))} for d, v in sorted(by.items())}


def root_guess_baseline(exs):
    """Expected choice accuracy of guessing uniformly among the root values in the program
    (the best a model can do without following the chain). Returns {depth: acc}."""
    by = defaultdict(list)
    for ex in exs:
        roots = [rhs for rhs in (ln.split("=")[1] for ln in ex.state.split(";")) if rhs.isdigit()]
        for qid, q in ex.questions.items():
            if q["type"] == "choice":
                by[ex.meta[qid]["depth"]].append(sum(int(r) == ex.labels[qid] for r in roots) / len(roots))
    return {d: float(np.mean(v)) for d, v in sorted(by.items())}


def summarize(rows):
    by = defaultdict(list)
    for d, t, s, h, c in rows:
        by[(d, s)].append(h)
        by[(d, s, t)].append(h)
    acc = defaultdict(dict)
    acc_type = defaultdict(lambda: defaultdict(dict))
    for k, v in by.items():
        if len(k) == 2:
            acc[k[1]][k[0]] = float(np.mean(v))
        else:
            acc_type[k[2]][k[1]][k[0]] = float(np.mean(v))
    cal = {}
    steps = sorted({r[2] for r in rows})
    for s in steps:
        c = np.array([r[4] for r in rows if r[2] == s])
        h = np.array([r[3] for r in rows if r[2] == s], dtype=float)
        cal[s] = {"ece": ece(c, h), "acc": float(h.mean()), "reliability": reliability(c, h)}
    return {"acc": acc, "acc_by_type": acc_type, "calibration": cal,
            "n": {d: sum(1 for r in rows if r[0] == d and r[2] == steps[0]) for d in sorted({r[0] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--depths", default="1-16")
    ap.add_argument("--n", type=int, default=250, help="examples per main-chain depth")
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="", help="suffix for the output file: eval<tag>.json")
    args = ap.parse_args()
    lo, hi = map(int, args.depths.split("-"))
    api = ZReason.load(args.run)
    train_depth = json.loads((Path(args.run) / "config.json").read_text())["train"]["train_depth"]
    exs = make_dataset(args.seed, args.n, range(lo, hi + 1))
    rows, auto_rows = run_eval(api, exs, args.max_steps)
    res = summarize(rows)
    if auto_rows:
        res["auto"] = summarize_auto(auto_rows)
    res["train_depth"] = train_depth
    res["root_guess_choice"] = root_guess_baseline(exs)
    (Path(args.run) / f"eval{args.tag}.json").write_text(json.dumps(res, indent=1))
    steps = sorted(res["acc"])
    print("depth " + " ".join(f"{s:>5}" for s in steps))
    for d in range(lo, hi + 1):
        print(f"{d:>5} " + " ".join(f"{res['acc'][s].get(d, float('nan')):5.2f}" for s in steps))
    print("ECE   " + " ".join(f"{res['calibration'][s]['ece']:5.3f}" for s in steps))
    if "auto" in res:
        print("steps='auto': depth -> mean halting step (accuracy)")
        print("  " + "  ".join(f"{d}:{v['mean_steps']:.1f}({v['acc']:.2f})" for d, v in res["auto"].items()))


if __name__ == "__main__":
    main()
