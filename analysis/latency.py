"""Wall-clock cost of latent steps.

For one request (one state, several questions), measures the time of the reasoning part
(vectors already computed) as a function of `steps`, and, for runs on frozen Laya vectors,
the one-off cost of encoding the state's sentences with Laya's encoder. Batch 1 is the
latency a single caller sees; the batched figure is throughput.

    python -m analysis.latency runs/nl_laya_s0 --task nl
    python -m analysis.latency runs/pw_s0 --task proofwriter
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from zreason.api import ZReason
from zreason.batching import collate, to_device

STEPS = [1, 2, 4, 8, 16, 32]


def requests(task: str, n: int):
    if task == "proofwriter":
        from zreason import proofwriter
        exs = proofwriter.load("depth-5", "test")[:n]
    else:
        from train import nl_lines
        from zreason.nl_tasks import make_nl_example
        rng = random.Random(0)
        exs = [nl_lines(make_nl_example(rng, 8, n_questions=4)) for _ in range(n)]
    return [{"state": e.state, "questions": dict(list(e.questions.items())[:4])} for e in exs]


@torch.no_grad()
def time_it(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


@torch.no_grad()
def encoder_ms(reqs, reps: int) -> float:
    """Laya's encoder over the sentences of one state, each sentence encoded alone (as in laya_features)."""
    from zreason.laya_head import load_zlaya

    model, tok = load_zlaya(units="tokens")
    enc = model.encoder.to(torch.bfloat16).eval()
    sents = reqs[0]["state"].split("\n") + [q["instructions"] for q in reqs[0]["questions"].values()]
    ids = [[tok.cls_token_id] + tok(s, add_special_tokens=False)["input_ids"] + [tok.sep_token_id] for s in sents]
    L = max(map(len, ids))
    x = torch.full((len(ids), L), tok.pad_token_id)
    for i, s in enumerate(ids):
        x[i, : len(s)] = torch.tensor(s)
    x = x.cuda()
    att = (x != tok.pad_token_id).long()

    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc(input_ids=x, attention_mask=att)
    return time_it(run, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", default="nl", choices=["nl", "proofwriter"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    api = ZReason.load(args.run)
    model = api.model.eval()
    reqs = requests(args.task, args.batch)
    one = to_device(collate(reqs[:1], api.tok), api.device)
    many = to_device(collate(reqs, api.tok), api.device)
    res = {"gpu": torch.cuda.get_device_name(), "questions_per_request": len(reqs[0]["questions"]),
           "batch": args.batch, "steps": {}}
    for s in STEPS:
        def single():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(one, steps=s)

        def batched():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(many, steps=s)
        b_ms = time_it(batched, max(5, args.reps // 5))
        res["steps"][s] = {"ms_per_request": time_it(single, args.reps),
                           "requests_per_s_batched": args.batch / b_ms * 1000}
    if model.cfg.ext_dim:
        res["laya_encoder_ms_per_request"] = encoder_ms(reqs, args.reps)
    (Path(args.run) / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
