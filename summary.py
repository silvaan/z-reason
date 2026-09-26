"""Markdown tables for REPORT.md, computed from runs/*/eval*.json.

    python summary.py --main runs/zr_s0 runs/zr_s1 runs/zr_s2 --fixed runs/fixed8_s0 --untied runs/untied8_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TYPES = {"choice": "0", "score": "1", "noul": "2"}


def load(run, tag=""):
    return json.loads((Path(run) / f"eval{tag}.json").read_text())


def acc(r, step, depth, metric="choice"):
    src = r["acc"] if metric == "all" else r["acc_by_type"][TYPES[metric]]
    return src.get(str(step), {}).get(str(depth), np.nan)


def fmt(xs):
    xs = np.array(xs, dtype=float)
    if len(xs) == 1:
        return f"{100 * xs[0]:.1f}"
    return f"{100 * xs.mean():.1f} ± {100 * xs.std():.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--fixed", default=None)
    ap.add_argument("--untied", default=None)
    args = ap.parse_args()
    mains = [load(m) for m in args.main]
    fixed = load(args.fixed) if args.fixed else None
    untied = load(args.untied) if args.untied else None
    steps = [1, 2, 4, 8, 16, 32, 128]
    groups = {"1-4": range(1, 5), "5-8 (max train)": range(5, 9), "9-12 (OOD)": range(9, 13), "13-16 (OOD)": range(13, 17)}

    print(f"### Choice accuracy (%) by depth group and test-time steps — mean ± std over {len(mains)} seeds\n")
    print("| depth | " + " | ".join(f"{s} steps" for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    for g, ds in groups.items():
        cells = [fmt([np.mean([acc(r, s, d) for d in ds]) for r in mains]) for s in steps]
        print(f"| {g} | " + " | ".join(cells) + " |")

    print("\n### Model comparison: choice accuracy (%) per depth group\n")
    print("| model | params | test-time compute | " + " | ".join(groups) + " |")
    print("|---|---|---|" + "---|" * len(groups))
    params = {m: json.loads((Path(m) / "config.json").read_text())["params"] for m in
              args.main[:1] + [x for x in (args.fixed, args.untied) if x]}
    rows = [(f"z-reason (random steps), 8 steps", params[args.main[0]], "8 steps", mains, 8),
            (f"z-reason (random steps), 32 steps", params[args.main[0]], "32 steps", mains, 32)]
    if fixed:
        rows += [("same model, trained with 8 fixed steps, 8 steps", params[args.fixed], "8 steps", [fixed], 8),
                 ("same model, trained with 8 fixed steps, 32 steps", params[args.fixed], "32 steps", [fixed], 32)]
    if untied:
        rows += [("untied deep transformer (8 × 2 blocks)", params[args.untied], "fixed (= 8 steps)", [untied], 8)]
    for name, p, comp, rs, s in rows:
        cells = [fmt([np.mean([acc(r, s, d) for d in ds]) for r in rs]) for ds in groups.values()]
        print(f"| {name} | {p / 1e6:.1f}M | {comp} | " + " | ".join(cells) + " |")

    print("\n### Calibration (ECE, all question types)\n")
    print("| steps | " + " | ".join(str(s) for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    print("| z-reason | " + " | ".join(f"{np.mean([r['calibration'][str(s)]['ece'] for r in mains]):.3f}" for s in steps) + " |")
    if fixed:
        print("| fixed-8 | " + " | ".join(f"{fixed['calibration'][str(s)]['ece']:.3f}" for s in steps) + " |")

    print('\n### steps="auto" (halt when no answer probability moves by more than 1e-3)\n')
    ds = [1, 2, 4, 8, 12, 16]
    print("| depth | " + " | ".join(str(d) for d in ds) + " |")
    print("|---|" + "---|" * len(ds))
    print("| mean steps used | " + " | ".join(f"{np.mean([r['auto'][str(d)]['mean_steps'] for r in mains]):.1f}" for d in ds) + " |")
    print("| accuracy (all types, %) | " + " | ".join(f"{100 * np.mean([r['auto'][str(d)]['acc'] for r in mains]):.1f}" for d in ds) + " |")

    deep = [Path(m) / "eval_deep.json" for m in args.main]
    if all(p.exists() for p in deep):
        dm = [json.loads(p.read_text()) for p in deep]
        rg = dm[0]["root_guess_choice"]
        dds = [16, 20, 24, 28, 32]
        print("\n### Deep extrapolation: choice accuracy (%), best of {16, 32, 64} steps vs. the root-guess shortcut\n")
        print("| depth | " + " | ".join(str(d) for d in dds) + " |")
        print("|---|" + "---|" * len(dds))
        best = [fmt([max(acc(r, s, d) for s in (16, 32, 64)) for r in dm]) for d in dds]
        print("| z-reason | " + " | ".join(best) + " |")
        for name, run in (("fixed-8", args.fixed), ("untied", args.untied)):
            p = Path(run) / "eval_deep.json" if run else None
            if p and p.exists():
                r = json.loads(p.read_text())
                ss = [8] if name == "untied" else [16, 32, 64]
                print(f"| {name} | " + " | ".join(fmt([max(acc(r, s, d) for s in ss)]) for d in dds) + " |")
        print("| root-guess shortcut | " + " | ".join(f"{100 * rg[str(d)]:.1f}" for d in dds) + " |")


if __name__ == "__main__":
    main()
