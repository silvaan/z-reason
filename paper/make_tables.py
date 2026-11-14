"""LaTeX tables for the paper, computed from runs/*/eval*.json (run from the repo root).

    python paper/make_tables.py --main runs/zr_s0 runs/zr_s1 runs/zr_s2 \
        --fixed runs/fixed8_s0 --untied runs/untied8_s0 \
        --nl runs/nl_laya_s0 runs/nl_laya_s1 runs/nl_laya_s2 --nl-fixed1 runs/nl_laya_fixed1_s0 \
        --pw runs/pw_s0 runs/pw_s1 runs/pw_s2 --pw-fixed1 runs/pw_fixed1_s0 \
        --pw-fixed8 runs/pw_fixed8_s0 --pw-untied runs/pw_untied8_s0 --out paper/tables
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


# ---------------------------------------------------------------- v2: natural language

NL_GROUPS = {"1 to 4": range(1, 5), "5 to 8": range(5, 9), "9 to 16$^\\dagger$": range(9, 17),
             "17 to 24$^\\dagger$": range(17, 25)}


def table_nl(mains, fixed1, out):
    """Choice accuracy with seen wording, by depth group and steps; one-pass baseline and root guess."""
    steps = [1, 2, 4, 8, 16, 32]
    rg = mains[0]["root_guess_choice"]
    n = len(steps)
    lines = [r"\begin{tabular}{l" + "c" * n + "cc}", r"\toprule",
             r"& \multicolumn{" + str(n) + r"}{c}{\zr{}, test-time steps} & one & root \\",
             r"\cmidrule(lr){2-" + str(n + 1) + "}",
             "depth & " + " & ".join(map(str, steps)) + r" & pass & guess \\", r"\midrule"]
    for g, ds in NL_GROUPS.items():
        vals = [[np.nanmean([acc(r, s, d) for d in ds]) for r in mains] for s in steps]
        one = cell([np.nanmean([acc(fixed1, 1, d) for d in ds])]) if fixed1 else "--"
        guess = 100 * np.mean([rg[str(d)] for d in ds if str(d) in rg])
        lines.append(g + " & " + " & ".join(cell(v) for v in vals) + f" & {one} & {guess:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "nl_steps.tex").write_text("\n".join(lines) + "\n")


def table_nl_wording(mains, out):
    """Held-out templates, one group at a time (analysis/heldout_ablation.py), all question types, depths 1-8."""
    abl = [json.loads((Path(r) / "heldout_ablation.json").read_text()) for r in mains]
    names = {"none": "none (seen wording)", "root": "value sentences", "copy": "copy sentences",
             "question": "questions", "all": "all"}
    steps = ["1", "4", "16"]
    lines = [r"\begin{tabular}{l" + "c" * len(steps) + "}", r"\toprule",
             r"held-out templates & \multicolumn{" + str(len(steps)) + r"}{c}{test-time steps} \\",
             r"\cmidrule(lr){2-" + str(len(steps) + 1) + "}",
             " & " + " & ".join(steps) + r" \\", r"\midrule"]
    for k, name in names.items():
        lines.append(name + " & " + " & ".join(cell([a[k][s] for a in abl]) for s in steps) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "nl_wording.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- v2: ProofWriter

PW_DEPTHS = [0, 1, 2, 3, 4, 5]


def pw_acc(r, step, depth):
    """Accuracy at a proof depth; depth 5 also holds the few Unknown questions labelled deeper."""
    a, n = r["acc"][str(step)], r["n"]
    ds = [d for d in a if (int(d) >= 5 if depth == 5 else int(d) == depth)]
    return sum(a[d] * n[d] for d in ds) / sum(n[d] for d in ds)


def pw_overall(r, step):
    a, n = r["acc"][str(step)], r["n"]
    return sum(a[d] * n[d] for d in a) / sum(n[d] for d in a)


def pw_majority(r, depth=None):
    m, n = r["majority_by_depth"], r["n"]
    ds = [d for d in m if depth is None or (int(d) >= 5 if depth == 5 else int(d) == depth)]
    return sum(m[d] * n[d] for d in ds) / sum(n[d] for d in ds)


def table_pw(mains, fixed1, fixed8, untied, out):
    load_pw = lambda r, c: json.loads((Path(r) / f"eval_pw_{c}.json").read_text())
    d5 = [load_pw(r, "depth-5") for r in mains]
    cols = " & ".join(str(d) if d <= 3 else f"{d}$^\\dagger$" for d in PW_DEPTHS)
    lines = [r"\begin{tabular}{llc" + "c" * len(PW_DEPTHS) + "c}", r"\toprule",
             r"model & train steps & test steps & " + cols + r" & all \\", r"\midrule"]

    def row(name, tr, rs, s):
        cells = [cell([pw_acc(r, s, d) for r in rs]) for d in PW_DEPTHS]
        return f"{name} & {tr} & {s} & " + " & ".join(cells) + " & " + cell([pw_overall(r, s) for r in rs]) + r" \\"
    for s in [1, 2, 4, 16]:
        lines.append(row("\\zr{}", "random", d5, s))
    if fixed1:
        lines.append(row("one pass", "1", [load_pw(fixed1, "depth-5")], 1))
    if fixed8:
        f8 = [load_pw(fixed8, "depth-5")]
        lines += [row("\\zr{}", "fixed 8", f8, 8), row("\\zr{}", "fixed 8", f8, 16)]
    if untied:
        lines.append(row("untied transformer", "none", [load_pw(untied, "depth-5")], 8))
    lines.append(r"\midrule")
    lines.append("majority answer & & & " + " & ".join(f"{100 * pw_majority(d5[0], d):.1f}" for d in PW_DEPTHS)
                 + f" & {100 * pw_majority(d5[0]):.1f}" + r" \\")
    lines.append("questions & & & " + " & ".join(f"{sum(v for k, v in d5[0]['n'].items() if (int(k) >= 5 if d == 5 else int(k) == d)):,}"
                                                 for d in PW_DEPTHS) + f" & {sum(d5[0]['n'].values()):,}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "pw_depth.tex").write_text("\n".join(lines) + "\n")

    # other test sets, overall accuracy
    sets = [("depth-3", "D3 test"), ("depth-5", "D5 test"), ("birds-electricity", "Birds-Electricity"), ("NatLang", "ParaRules")]
    lines = [r"\begin{tabular}{llc" + "c" * len(sets) + "}", r"\toprule",
             "model & train steps & test steps & " + " & ".join(n for _, n in sets) + r" \\", r"\midrule"]

    def row2(name, tr, runs, s):
        return f"{name} & {tr} & {s} & " + " & ".join(cell([pw_overall(load_pw(r, c), s) for r in runs]) for c, _ in sets) + r" \\"
    lines += [row2("\\zr{}", "random", mains, 1), row2("\\zr{}", "random", mains, 16)]
    if fixed1:
        lines.append(row2("one pass", "1", [fixed1], 1))
    if untied:
        lines.append(row2("untied transformer", "none", [untied], 8))
    lines.append(r"\midrule")
    lines.append("majority answer & & & " + " & ".join(f"{100 * pw_majority(load_pw(mains[0], c)):.1f}" for c, _ in sets) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "pw_transfer.tex").write_text("\n".join(lines) + "\n")


def table_pw_confusion(mains, fixed1, out, step=16):
    """What deep questions are answered with: rates by true answer, depths 3-5 (analysis/pw_confusion.py)."""
    def conf(r, s):
        return np.array(json.loads((Path(r) / "confusion_depth-5.json").read_text())["confusion"][str(s)])
    z = sum(conf(r, step) for r in mains)
    models = [("\\zr{}, " + str(step) + " steps", z)]
    if fixed1:
        models.append(("one pass", conf(fixed1, 1)))
    ds = [3, 4, 5]
    lines = [r"\begin{tabular}{l" + "c" * (2 * len(ds)) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{2}{c}{depth " + (str(d) if d <= 3 else f"{d}$^\\dagger$") + "}" for d in ds) + r" \\",
             " ".join(r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}" for i in range(len(ds))),
             " & " + " & ".join(r"\zr{} & one pass" for _ in ds) + r" \\", r"\midrule"]

    def pct(x):
        return f"{100 * x:.1f}"
    rows = [("True/False answered correctly", lambda c: (c[0, 0] + c[1, 1]) / c[:2].sum()),
            ("True/False answered Unknown", lambda c: (c[0, 2] + c[1, 2]) / c[:2].sum()),
            ("True/False answered with the opposite", lambda c: (c[0, 1] + c[1, 0]) / c[:2].sum()),
            ("Unknown answered correctly", lambda c: c[2, 2] / c[2].sum()),
            ("True/False predictions that are right", lambda c: (c[0, 0] + c[1, 1]) / c[:, :2].sum())]
    for name, f in rows:
        lines.append(name + " & " + " & ".join(" & ".join(pct(f(m[d])) for _, m in models) for d in ds) + r" \\")
    lines.append(r"\midrule")
    n = conf(mains[0], step)  # the same questions for every model
    lines.append("questions & " + " & ".join(r"\multicolumn{2}{c}{" + f"{int(n[d].sum()):,}" + "}" for d in ds) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "pw_confusion.tex").write_text("\n".join(lines) + "\n")


def table_latency(nl_run, pw_run, out):
    ln = json.loads((Path(nl_run) / "latency.json").read_text())
    lp = json.loads((Path(pw_run) / "latency.json").read_text())
    steps = [s for s in ln["steps"] if s in lp["steps"]]
    lines = [r"\begin{tabular}{l" + "c" * len(steps) + "}", r"\toprule",
             "test-time steps & " + " & ".join(steps) + r" \\", r"\midrule",
             "variable tracking, ms per request & " + " & ".join(f"{ln['steps'][s]['ms_per_request']:.1f}" for s in steps) + r" \\",
             "ProofWriter, ms per request & " + " & ".join(f"{lp['steps'][s]['ms_per_request']:.1f}" for s in steps) + r" \\",
             "variable tracking, requests per second in batches of " + str(ln["batch"]) + " & "
             + " & ".join(f"{ln['steps'][s]['requests_per_s_batched']:,.0f}" for s in steps) + r" \\",
             "ProofWriter, requests per second in batches of " + str(lp["batch"]) + " & "
             + " & ".join(f"{lp['steps'][s]['requests_per_s_batched']:,.0f}" for s in steps) + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    (out / "latency.tex").write_text("\n".join(lines) + "\n")
    print(f"laya encoder ms per request: {ln.get('laya_encoder_ms_per_request')}, gpu {ln['gpu']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--fixed")
    ap.add_argument("--untied")
    ap.add_argument("--nl", nargs="*", default=[])
    ap.add_argument("--nl-fixed1")
    ap.add_argument("--pw", nargs="*", default=[])
    ap.add_argument("--pw-fixed1")
    ap.add_argument("--pw-fixed8")
    ap.add_argument("--pw-untied")
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
    if args.nl:
        nl = [load(r, "_nl_train") for r in args.nl]
        table_nl(nl, load(args.nl_fixed1, "_nl_train") if args.nl_fixed1 else None, out)
        table_nl_wording(args.nl, out)
    if args.pw:
        table_pw(args.pw, args.pw_fixed1, args.pw_fixed8, args.pw_untied, out)
        table_pw_confusion(args.pw, args.pw_fixed1, out)
    if args.nl and args.pw:
        table_latency(args.nl[0], args.pw[0], out)
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
