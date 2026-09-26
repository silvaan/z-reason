"""Train a ZReasoner on the synthetic variable-chain task.

Example:
    python train.py --out runs/zr_s0 --seed 0
    python train.py --out runs/fixed8_s0 --fixed-steps 8        # baseline: no random steps
    python train.py --out runs/untied8_s0 --untied 8            # baseline: standard deep transformer
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import torch.multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from zreason.batching import collate, to_device
from zreason.model import Block, ZConfig, ZReasoner, correct, question_loss
from zreason.tasks import make_chain_example, make_dataset
from zreason.tokenizer import CharTokenizer


class ChainStream(IterableDataset):
    """Infinite stream of collated batches with an adaptive curriculum.

    `level` in [0, 1] (shared with the training loop) sets the program size
    (2 -> 40 variables) and the maximum chain depth (1 -> max_depth). The loop
    raises it only once the model answers the current level well. Without a
    curriculum, finding the right line among many (associative recall) sits on a
    loss plateau whose escape is seed-dependent.
    """

    def __init__(self, seed, batch_size, max_depth, n_questions, level):
        self.seed, self.bs, self.max_depth, self.nq, self.level = seed, batch_size, max_depth, n_questions, level
        self.tok = CharTokenizer()

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            f = self.level.value
            max_vars = 2 + round(f * 38)
            dmax = max(1, round(f * self.max_depth))
            exs = []
            for _ in range(self.bs):
                d = rng.randint(1, dmax)
                exs.append(make_chain_example(rng, d, n_questions=self.nq,
                                              min_vars=min(24, max(2, max_vars // 2)), max_vars=max(max_vars, d + 1)))
            yield collate([{"state": e.state, "questions": e.questions} for e in exs], self.tok,
                          labels=[e.labels for e in exs])


def sample_steps(rng: np.random.Generator, mean: float, max_steps: int, sigma: float = 0.5) -> int:
    """Huginn-style heavy-tailed schedule: r = 1 + Poisson(lambda), lambda ~ LogNormal."""
    lam = rng.lognormal(math.log(max(mean - 1, 1e-3)) - sigma**2 / 2, sigma)
    return int(min(max(1 + rng.poisson(lam), 1), max_steps))


@torch.no_grad()
def evaluate(model, exs, tok, steps, device, bs=256):
    model.eval()
    hits, n = 0, 0
    for i in range(0, len(exs), bs):
        chunk = exs[i: i + bs]
        b = to_device(collate([{"state": e.state, "questions": e.questions} for e in chunk], tok,
                              labels=[e.labels for e in chunk]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, steps=steps, generator=torch.Generator(device=device).manual_seed(0))
        c = correct(out, b)
        hits += c.sum().item()
        n += c.numel()
    model.train()
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-core", type=int, default=2)
    ap.add_argument("--train-depth", type=int, default=8)
    ap.add_argument("--n-questions", type=int, default=4)
    ap.add_argument("--steps-mean", type=float, default=12)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--grad-steps", type=int, default=8)
    ap.add_argument("--fixed-steps", type=int, default=0)
    ap.add_argument("--untied", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--curriculum", type=float, default=0.8,
                    help="raise the curriculum level when EMA choice accuracy exceeds this (0 = no curriculum)")
    ap.add_argument("--level-step", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    tok = CharTokenizer()

    cfg = ZConfig(vocab_size=tok.vocab_size, d=args.d, heads=8, n_core=args.n_core, untied_steps=args.untied)
    model = ZReasoner(cfg).to(device)
    for mod in model.modules():
        if isinstance(mod, Block):
            mod.compile(dynamic=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M", flush=True)
    (out / "config.json").write_text(json.dumps({"model": asdict(cfg), "train": vars(args), "params": n_params}, indent=2))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=args.lr, betas=(0.9, 0.95))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i / args.iters, 1))))

    level = mp.Value("d", 0.0 if args.curriculum else 1.0)
    loader = DataLoader(ChainStream(args.seed, args.bs, args.train_depth, args.n_questions, level), batch_size=None,
                        num_workers=args.workers, prefetch_factor=4, persistent_workers=True)
    val = make_dataset(10_000 + args.seed, 64, range(1, args.train_depth + 1))
    log = open(out / "log.jsonl", "w")
    t0 = time.time()
    model.train()
    ema, last_raise = 0.0, 0
    for it, batch in enumerate(loader):
        if it >= args.iters:
            break
        batch = to_device(batch, device)
        steps = args.fixed_steps or sample_steps(np_rng, args.steps_mean, args.max_steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(batch, steps=steps, grad_steps=args.grad_steps)
            loss, stats = question_loss(o, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            ch = batch["q_type"] == 0
            if ch.any():
                ema = 0.98 * ema + 0.02 * correct(o, batch)[ch].float().mean().item()
        if args.curriculum and level.value < 1 and it - last_raise >= 200 and ema > args.curriculum:
            level.value = min(1.0, level.value + args.level_step)
            last_raise, ema = it, 0.0
            print(json.dumps({"it": it, "level": level.value}), flush=True)
        if it % 100 == 0:
            acc = correct(o, batch).float().mean().item()
            rec = {"it": it, "loss": loss.item(), "acc": acc, "choice_ema": ema, "level": level.value, "steps": steps, "gn": gn.item(),
                   "lr": sched.get_last_lr()[0], "t": time.time() - t0, **stats}
            if it % args.eval_every == 0 or it == args.iters - 1:
                ss = [args.untied] if args.untied else [1, 4, 8, 16, 32]
                rec["val"] = {s: evaluate(model, val, tok, s, device) for s in ss}
                torch.save(model.state_dict(), out / "model.pt")
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save(model.state_dict(), out / "model.pt")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
