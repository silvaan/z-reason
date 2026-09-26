"""LaTeX tables for the paper, computed from runs/*/eval*.json (run from the repo root).

    python paper/make_tables.py --main runs/zr_s0 runs/zr_s1 runs/zr_s2 \
        --fixed runs/fixed8_s0 --untied runs/untied8_s0 --out paper/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CH = "0"  # choice


def load(run, tag=""):
    return json.loads((Path(run) / f"eval{tag}.json").read_text())


def acc(r, step, depth, t=CH):
    src = r["acc"] if t == "all" else r["acc_by_type"][t]
    return src.get(str(step), {}).get(str(depth), np.nan)


def cell(xs, bold=False):
    xs = np.array(xs, dtype=float) * 100
    s = f"{xs.mean():.1f}" if len(xs) == 1 else f"{xs.mean():.1f}\\,$\\pm$\\,{xs.std():.1f}"
    return f"\\textbf{{{s}}}" if bold else s


GROUPS = {"1 to 4": range(1, 5), "5 to 8": range(5, 9), "9 to 12$^\\dagger$": range(9, 13), "13 to 16$^\\dagger$": range(13, 17)}


def table_steps(mains, out):
    steps = [1, 2, 3, 4, 6, 8, 16, 128]
    rg = mains[0].get("root_guess_choice", {})
    n = len(steps)
    lines = [r"\begin{tabular}{l" + "c" * n + "c}", r"\toprule",
             r"& \multicolumn{" + str(n) + r"}{c}{test-time steps} & root \\", r"\cmidrule(lr){2-" + str(n + 1) + "}",
             "depth & " + " & ".join(map(str, steps)) + r" & guess \\", r"\midrule"]
    for g, ds in GROUPS.items():
        vals = [[np.mean([acc(r, s, d) for d in ds]) for r in mains] for s in steps]
        guess = 100 * np.mean([rg[str(d)] for d in ds])
        lines.append(g + " & " + " & ".join(cell(v) for v in vals) + f" & {guess:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "steps.tex").write_text("\n".join(lines) + "\n")


def table_compare(mains, fixed, untied, params, deep, out):
    dds = [16, 20, 24, 28, 32]
    lines = [r"\begin{tabular}{llcc" + "c" * len(GROUPS) + "}", r"\toprule",
             r"model & train steps & params & test steps & " + " & ".join(GROUPS) + r" \\", r"\midrule"]
    rows = [("z-reason", "random", params["main"], 8, mains),
            ("z-reason", "random", params["main"], 32, mains)]
    if fixed:
        rows += [("z-reason", "fixed 8", params["fixed"], 8, [fixed]), ("z-reason", "fixed 8", params["fixed"], 32, [fixed])]
    if untied:
        rows += [("untied transformer", "none", params["untied"], "8", [untied])]
    for name, tr, p, s, rs in rows:
        ss = 8 if s == "8" else s
        cells = [cell([np.mean([acc(r, ss, d) for d in ds]) for r in rs]) for ds in GROUPS.values()]
        lines.append(f"{name} & {tr} & {p / 1e6:.1f}M & {s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "compare.tex").write_text("\n".join(lines) + "\n")

    if deep:
        dm, df, du = deep
        lines = [r"\begin{tabular}{l" + "c" * len(dds) + "}", r"\toprule",
                 "depth & " + " & ".join(map(str, dds)) + r" \\", r"\midrule"]
        lines.append("z-reason, random steps, 32 steps & " + " & ".join(cell([acc(r, 32, d) for r in dm]) for d in dds) + r" \\")
        if df:
            lines.append("z-reason, fixed 8, 32 steps & " + " & ".join(cell([acc(df, 32, d)]) for d in dds) + r" \\")
        if du:
            lines.append("untied transformer & " + " & ".join(cell([acc(du, 8, d)]) for d in dds) + r" \\")
        rg = dm[0]["root_guess_choice"]
        lines.append(r"\midrule")
        lines.append("root-guess shortcut & " + " & ".join(f"{100 * rg[str(d)]:.1f}" for d in dds) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "deep.tex").write_text("\n".join(lines) + "\n")


def table_calibration(mains, fixed, out):
    steps = [1, 2, 4, 8, 16, 128]
    lines = [r"\begin{tabular}{l" + "c" * len(steps) + "}", r"\toprule",
             "test-time steps & " + " & ".join(map(str, steps)) + r" \\", r"\midrule"]

    def ece(r, s):
        return r["calibration"][str(s)]["ece"]

    def accall(r, s):
        return r["calibration"][str(s)]["acc"]

    lines.append("random steps, accuracy in \\% & " + " & ".join(cell([accall(r, s) for r in mains]) for s in steps) + r" \\")
    lines.append("random steps, ECE & " + " & ".join(
        f"{np.mean([ece(r, s) for r in mains]):.3f}" for s in steps) + r" \\")
    if fixed:
        lines.append("fixed 8, accuracy in \\% & " + " & ".join(cell([accall(fixed, s)]) for s in steps) + r" \\")
        lines.append("fixed 8, ECE & " + " & ".join(f"{ece(fixed, s):.3f}" for s in steps) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "calibration.tex").write_text("\n".join(lines) + "\n")


def table_auto(mains, out):
    ds = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    lines = [r"\begin{tabular}{l" + "c" * len(ds) + "}", r"\toprule",
             "depth & " + " & ".join(map(str, ds)) + r" \\", r"\midrule",
             "mean steps used & " + " & ".join(f"{np.mean([r['auto'][str(d)]['mean_steps'] for r in mains]):.1f}" for d in ds) + r" \\",
             "accuracy in \\% & " + " & ".join(f"{100 * np.mean([r['auto'][str(d)]['acc'] for r in mains]):.1f}" for d in ds) + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    (out / "auto.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--fixed")
    ap.add_argument("--untied")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mains = [load(m) for m in args.main]
    fixed = load(args.fixed) if args.fixed else None
    untied = load(args.untied) if args.untied else None
    cfg = lambda r: json.loads((Path(r) / "config.json").read_text())["params"]
    params = {"main": cfg(args.main[0]), "fixed": cfg(args.fixed) if args.fixed else 0,
              "untied": cfg(args.untied) if args.untied else 0}
    deep = None
    if all((Path(m) / "eval_deep.json").exists() for m in args.main):
        deep = ([load(m, "_deep") for m in args.main],
                load(args.fixed, "_deep") if args.fixed and (Path(args.fixed) / "eval_deep.json").exists() else None,
                load(args.untied, "_deep") if args.untied and (Path(args.untied) / "eval_deep.json").exists() else None)
    table_steps(mains, out)
    table_compare(mains, fixed, untied, params, deep, out)
    table_calibration(mains, fixed, out)
    table_auto(mains, out)
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
