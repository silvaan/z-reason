"""Train z-Laya (z-reason on top of Laya's frozen encoder) on the NL variable tracking task.

Data: by default every fact and question is encoded once on its own (zreason/laya_tables.py)
and batches are drawn fresh at every iteration. `--data cache` instead reads Laya's joint
encoding of question and state from cache_laya.py, the only option for `--units tokens`.

Example:
    python train_laya.py --out runs/zl_s0 --seed 0                  # random steps
    python train_laya.py --out runs/zl_fixed1_s0 --fixed-steps 1     # one pass, like Jev / Laya
    python train_laya.py --out runs/zl_untied8_s0 --untied 8         # 8 distinct heads, same compute as 8 steps
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from train import sample_steps
from zreason.laya_head import QTYPES, add_facts, build_sequence, collate, fact_spans, load_zlaya, question_loss
from zreason.laya_tables import Tables, build_tables
from zreason.nl_tasks import make_nl_dataset, make_nl_example


class CachedStream(IterableDataset):
    """Batches of cached encoder states. `level` (shared with the loop) caps the depth of the
    questions drawn, so the curriculum raises the difficulty as in v1."""

    def __init__(self, cache, seed, batch_size, max_depth, level, facts=None):
        self.cache, self.seed, self.bs, self.max_depth, self.level = Path(cache), seed, batch_size, max_depth, level
        self.facts = facts  # (sep id, period id) for units="facts"

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (wi.id if wi else 0))
        ix = torch.load(self.cache / "index.pt", weights_only=False)
        states = np.load(self.cache / "states.npy", mmap_mode="r")
        depth = np.array(ix["depth"])
        by_cap = {c: np.flatnonzero(depth <= c) for c in range(self.max_depth + 1)}
        while True:
            cap = max(1, round(self.level.value * self.max_depth))
            idx = rng.choice(by_cap[cap], self.bs)
            off = ix["offsets"]
            b = collate([(ix["ids"][j], ix["markers"][j]) for j in idx], [ix["qtype"][j] for j in idx], 0,
                        labels=[ix["label"][j] for j in idx])
            if self.facts:
                add_facts(b, *self.facts)
            h = torch.zeros(len(idx), b["input_ids"].shape[1], states.shape[1], dtype=torch.float16)
            for r, j in enumerate(idx):
                h[r, : off[j + 1] - off[j]] = torch.from_numpy(np.array(states[off[j]: off[j + 1]]))
            b["h"] = h
            yield b


def table_stream(tables, seed, batch_size, n_questions, max_depth, level, max_vars=28):
    """Fresh examples at every iteration. As in v1, `level` in [0, 1] sets both the number of
    people (2 -> max_vars) and the depth cap (1 -> max_depth): finding the right fact among many
    is itself hard to learn and has to start from small states."""
    rng = random.Random(seed)
    while True:
        f = level.value
        n_max = 2 + round(f * (max_vars - 2))
        dmax = max(1, round(f * max_depth))
        exs = []
        for _ in range(batch_size // n_questions):
            d = rng.randint(1, dmax)
            exs.append(make_nl_example(rng, d, n_questions=n_questions, min_vars=min(16, max(2, n_max // 2)),
                                       max_vars=max(n_max, d + 1)))
        yield tables.batch(exs)


def load_tables(model, tok, path="cache/nl_tables.pt"):
    if not Path(path).exists():
        Path(path).parent.mkdir(exist_ok=True)
        build_tables(model, tok, path)
    return Tables(path)


def fit_fact_norm(model, cache, n=200, seed=0):
    """Per-dimension mean and std of the state tokens, from cached encoder states."""
    ix = torch.load(Path(cache) / "index.pt", weights_only=False)
    states = np.load(Path(cache) / "states.npy", mmap_mode="r")
    off = ix["offsets"]
    toks = []
    for j in np.random.default_rng(seed).choice(len(ix["ids"]), n, replace=False):
        hl, spans = fact_spans(ix["ids"][j], model.sep_id, model.period_id)
        toks.append(np.asarray(states[off[j] + spans[0][0]: off[j] + spans[-1][1]], dtype=np.float32))
    t = torch.from_numpy(np.concatenate(toks))
    model.fact_mu.copy_(t.mean(0))
    model.fact_sd.copy_(t.std(0) + 1e-5)


def encode_eval(model, tok, exs, bs=48):
    """Flatten examples into per-question sequences with their encoder states (kept on CPU)."""
    seqs, qt, lab, dep = [], [], [], []
    for e in exs:
        for qid, q in e.questions.items():
            seqs.append(build_sequence(tok, e.state, q))
            qt.append(QTYPES[q["type"]]); lab.append(e.labels[qid]); dep.append(e.meta[qid]["depth"])
    batches = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(seqs), bs):
            b = collate(seqs[i: i + bs], qt[i: i + bs], tok.pad_token_id, labels=lab[i: i + bs])
            b["h"] = model.encode({k: v.cuda() for k, v in b.items()}).half().cpu()
            b["depth"] = torch.tensor(dep[i: i + bs])
            batches.append(b)
    return batches


@torch.no_grad()
def evaluate(model, batches, steps_list):
    """Accuracy per number of steps, from one traced pass of max(steps_list) iterations."""
    model.eval()
    hits = {s: 0 for s in steps_list}
    n = 0
    for b in batches:
        if model.units == "facts" and "fact_tok" not in b and "head_len" not in b:
            add_facts(b, model.sep_id, model.period_id)
        g = {k: v.cuda() for k, v in b.items()}
        h = g["h"].to(torch.bfloat16) if "h" in g else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(g, steps=max(steps_list), trace=True, h=h)
        for s in steps_list:
            lg = out["trace"][s - 1] if not model.untied_steps else out["logits"]
            hits[s] += (lg.argmax(-1) == g["label"]).sum().item()
        n += len(g["label"])
    model.train()
    model.encoder.eval()
    return {s: hits[s] / n for s in steps_list}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/nl_train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--train-depth", type=int, default=8)
    ap.add_argument("--steps-mean", type=float, default=6)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--grad-steps", type=int, default=6)
    ap.add_argument("--fixed-steps", type=int, default=0)
    ap.add_argument("--untied", type=int, default=0)
    ap.add_argument("--units", default="facts", choices=["facts", "tokens"])
    ap.add_argument("--data", default="tables", choices=["tables", "cache"])
    ap.add_argument("--n-questions", type=int, default=4, help="questions per state (tables)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--curriculum", type=float, default=0.8,
                    help="raise the depth cap when EMA accuracy exceeds this (0 = no curriculum)")
    ap.add_argument("--level-step", type=float, default=0.125)
    ap.add_argument("--full-depth-by", type=float, default=0.5,
                    help="the depth cap reaches its maximum by this fraction of training whatever the accuracy, "
                         "so a model that cannot pass the threshold still sees every depth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    model, tok = load_zlaya(untied_steps=args.untied, units=args.units, device=device)
    model.encoder.to(torch.bfloat16).eval()
    model.checkpointing = True
    assert args.data == "cache" or args.units == "facts", "--units tokens needs --data cache"
    if args.data == "tables":
        tables = load_tables(model, tok)
        mu, sd = tables.fact_stats()
        model.fact_mu.copy_(mu)
        model.fact_sd.copy_(sd)
    elif args.units == "facts":
        fit_fact_norm(model, args.cache)
    grad_steps = None if args.untied else args.grad_steps
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    print(f"trainable params: {n_train / 1e6:.1f}M", flush=True)
    (out / "config.json").write_text(json.dumps({"train": vars(args), "trainable_params": n_train}, indent=2))

    decay = [p for p in train_params if p.dim() >= 2]
    no_decay = [p for p in train_params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=args.lr, betas=(0.9, 0.95))
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i / args.iters, 1))))

    val_exs = make_nl_dataset(10_000 + args.seed, 32, range(1, args.train_depth + 1))
    level = mp.Value("d", (0.0 if args.data == "tables" else 1 / args.train_depth) if args.curriculum else 1.0)
    if args.data == "tables":
        val = [{k: v.cpu() for k, v in tables.batch(val_exs[i: i + 32]).items()} for i in range(0, len(val_exs), 32)]
        loader = table_stream(tables, args.seed, args.bs, args.n_questions, args.train_depth, level)
    else:
        val = encode_eval(model, tok, val_exs)
        loader = DataLoader(CachedStream(args.cache, args.seed, args.bs, args.train_depth, level,
                                         (model.sep_id, model.period_id) if args.units == "facts" else None),
                            batch_size=None, num_workers=args.workers, prefetch_factor=4, persistent_workers=True)
    log = open(out / "log.jsonl", "w")
    t0 = time.time()
    model.train()
    model.encoder.eval()
    ema, last_raise = 0.0, 0
    for it, b in enumerate(loader):
        if it >= args.iters:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        steps = args.fixed_steps or sample_steps(np_rng, args.steps_mean, args.max_steps)
        # no autocast weight cache: with tied weights it makes checkpoint recomputation save other tensors
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            o = model(b, steps=steps, grad_steps=grad_steps, h=b["h"].to(torch.bfloat16) if "h" in b else None)
            loss = question_loss(o["logits"], b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        acc = (o["logits"].argmax(-1) == b["label"]).float().mean().item()
        ema = 0.98 * ema + 0.02 * acc
        if args.curriculum and level.value < 1 and it - last_raise >= 200 and ema > args.curriculum:
            level.value = min(1.0, level.value + args.level_step)
            last_raise, ema = it, 0.0
            print(json.dumps({"it": it, "level": level.value}), flush=True)
        floor = math.ceil(args.train_depth * it / (args.full_depth_by * args.iters)) / args.train_depth if args.full_depth_by else 0
        if args.curriculum and level.value < min(floor, 1.0):
            level.value = min(floor, 1.0)
            print(json.dumps({"it": it, "level": level.value, "forced": True}), flush=True)
        if it % 50 == 0:
            rec = {"it": it, "loss": loss.item(), "acc": acc, "ema": ema, "level": level.value, "steps": steps,
                   "gn": gn.item(), "lr": sched.get_last_lr()[0], "gate": model.inject.abs().mean().item(),
                   "t": time.time() - t0}
            if it % args.eval_every == 0 or it == args.iters - 1:
                rec["val"] = evaluate(model, val, [args.untied] if args.untied else [1, 2, 4, 8, 16])
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
                           out / "head.pt")
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    rec = {"it": args.iters, "val": evaluate(model, val, [args.untied] if args.untied else [1, 2, 4, 8, 16])}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}, out / "head.pt")


if __name__ == "__main__":
    main()
