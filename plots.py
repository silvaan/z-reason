"""Paper figures from runs/*/eval.json.

    python plots.py --main runs/zr_s0 runs/zr_s1 runs/zr_s2 \
        --fixed runs/fixed8_s0 --untied runs/untied8_s0 --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# ordinal blue ramp (>= step 250 so the lightest mark clears 2:1 on the surface)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
})


METRIC = "choice"  # headline metric: choice accuracy (chance 10%, needs the full chain)


def load(run):
    r = json.loads((Path(run) / "eval.json").read_text())
    src = r["acc"] if METRIC == "all" else r["acc_by_type"][{"choice": "0", "score": "1", "noul": "2"}[METRIC]]
    acc = {int(s): {int(d): v for d, v in dd.items()} for s, dd in src.items()}
    return r, acc


def mean_acc(runs):
    """-> steps, depths, array (n_runs, n_steps, n_depths)."""
    accs = [load(r)[1] for r in runs]
    steps = sorted(accs[0])
    depths = sorted(accs[0][steps[0]])
    a = np.array([[[acc[s][d] for d in depths] for s in steps] for acc in accs])
    return steps, depths, a


def fig_steps_curves(runs, train_depth, out):
    steps, depths, a = mean_acc(runs)
    m, sd = a.mean(0), a.std(0)
    pick = [1, 4, 8, 10, 12, 16]
    pick = [d for d in pick if d in depths]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for c, d in zip(RAMP, pick):
        j = depths.index(d)
        ls = "-" if d <= train_depth else (0, (4, 2))
        ax.plot(steps, m[:, j], color=c, ls=ls, marker="o", ms=4, label=f"depth {d}" + ("" if d <= train_depth else ", unseen"))
        if len(runs) > 1:
            ax.fill_between(steps, m[:, j] - sd[:, j], m[:, j] + sd[:, j], color=c, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("latent reasoning steps at inference")
    ax.set_ylabel(f"{METRIC} accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("More latent steps solve harder problems")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_accuracy_vs_steps.png", dpi=200)
    plt.close(fig)


def fig_heatmap(runs, train_depth, out):
    steps, depths, a = mean_acc(runs)
    m = a.mean(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.grid(False)
    im = ax.imshow(m.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(steps)), [str(s) for s in steps])
    ax.set_yticks(range(len(depths)), [str(d) for d in depths])
    ax.axhline(depths.index(train_depth) + 0.5, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(len(steps) - 0.45, depths.index(train_depth) + 0.5, " max\n train\n depth", ha="left", va="center",
            fontsize=8, color=INK, clip_on=False)
    for i in range(len(steps)):
        for j in range(len(depths)):
            ax.text(i, j, f"{m[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if m[i, j] > 0.6 else INK)
    ax.set_xlabel("latent reasoning steps at inference")
    ax.set_ylabel("chain depth")
    ax.set_title(f"{METRIC.capitalize()} accuracy in % by chain depth and test-time steps")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.14)
    fig.tight_layout()
    fig.savefig(out / "fig2_heatmap.png", dpi=200)
    plt.close(fig)


def fig_compare(main, fixed, untied, train_depth, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    series = []
    steps, depths, a = mean_acc(main)
    for s, c in [(8, SERIES[0]), (max(steps), SERIES[1])]:
        series.append((f"z-reason, {s} steps", c, a[:, steps.index(s), :], "-"))
    if fixed:
        fs, fd, fa = mean_acc(fixed)
        best = int(np.argmax(fa.mean(0)[:, [fd.index(d) for d in fd if d <= train_depth]].mean(-1)))
        series.append((f"fixed-8 training, eval {fs[best]} steps", SERIES[2], fa[:, best, :], "-"))
    if untied:
        us, ud, ua = mean_acc(untied)
        series.append(("untied transformer, 16 blocks", SERIES[3], ua[:, 0, :], "-"))
    for name, c, arr, ls in series:
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(depths, mu, color=c, marker="o", ms=4, ls=ls, label=name)
        if arr.shape[0] > 1:
            ax.fill_between(depths, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    rg = load(main[0])[0].get("root_guess_choice")
    if rg and METRIC == "choice":
        xs = [d for d in depths if str(d) in rg]
        ax.plot(xs, [rg[str(d)] for d in xs], color=INK2, lw=1.2, ls=(0, (3, 2)), label="shortcut: guess a root value")
    ax.axvspan(train_depth + 0.5, max(depths) + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(train_depth + 0.7, 0.03, "unseen depths", fontsize=8, color=INK2)
    ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("chain depth")
    ax.set_ylabel(f"{METRIC} accuracy")
    ax.set_title("Recurrent latent reasoning vs. fixed-depth baselines")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_baselines.png", dpi=200)
    plt.close(fig)


def fig_calibration(main, out):
    r, _ = load(main[0])
    cal = {int(s): v for s, v in r["calibration"].items()}
    steps = sorted(cal)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    eces = np.array([[load(m)[0]["calibration"][str(s)]["ece"] for s in steps] for m in main])
    axs[0].plot(steps, eces.mean(0), color=SERIES[0], marker="o", ms=4)
    axs[0].set_xscale("log", base=2)
    ticks = [s for s in steps if s & (s - 1) == 0]  # powers of two
    axs[0].set_xticks(ticks, [str(s) for s in ticks])
    axs[0].set_xlabel("latent reasoning steps")
    axs[0].set_ylabel("expected calibration error")
    axs[0].set_title("Calibration across steps")
    axs[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 2)))
    for s, c in [(2, RAMP[0]), (8, RAMP[2]), (max(steps), RAMP[4])]:
        rel = [b for b in cal[s]["reliability"] if b["n"] > 20]
        axs[1].plot([b["conf"] for b in rel], [b["acc"] for b in rel], color=c, marker="o", ms=4,
                    label=f"{s} steps, ECE {cal[s]['ece']:.3f}")
    axs[1].set_xlabel("confidence")
    axs[1].set_ylabel("empirical accuracy")
    axs[1].set_title("Reliability diagram")
    axs[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_calibration.png", dpi=200)
    plt.close(fig)


def fig_auto(main, train_depth, out):
    runs = [load(m)[0] for m in main]
    if not all("auto" in r for r in runs):
        return
    depths = sorted(int(d) for d in runs[0]["auto"])
    st = np.array([[r["auto"][str(d)]["mean_steps"] for d in depths] for r in runs])
    ac = np.array([[r["auto"][str(d)]["acc"] for d in depths] for r in runs])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].plot(depths, st.mean(0), color=SERIES[0], marker="o", ms=4)
    if len(runs) > 1:
        axs[0].fill_between(depths, st.mean(0) - st.std(0), st.mean(0) + st.std(0), color=SERIES[0], alpha=0.15, lw=0)
    axs[0].set_xlabel("chain depth")
    axs[0].set_ylabel("mean steps used")
    axs[0].set_title('steps="auto": compute grows with difficulty')
    axs[1].plot(depths, ac.mean(0), color=SERIES[0], marker="o", ms=4)
    axs[1].set_ylim(0, 1.02)
    axs[1].set_xlabel("chain depth")
    axs[1].set_ylabel("accuracy, all question types")
    axs[1].set_title('Accuracy with steps="auto"')
    for ax in axs:
        ax.axvspan(train_depth + 0.5, max(depths) + 0.5, color=GRID, alpha=0.5, lw=0)
        ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
        ax.set_xticks(depths[::2])
    fig.tight_layout()
    fig.savefig(out / "fig5_auto_steps.png", dpi=200)
    plt.close(fig)


def steps_needed(runs, thr=0.99):
    """Minimal test-time steps reaching `thr` accuracy at each depth (mean over runs)."""
    steps, depths, a = mean_acc(runs)
    need = np.full((a.shape[0], len(depths)), np.nan)
    for r in range(a.shape[0]):
        for j in range(len(depths)):
            ok = [s for i, s in enumerate(steps) if a[r, i, j] >= thr]
            if ok:
                need[r, j] = ok[0]
    return depths, need


def fig_steps_needed(main, train_depth, out):
    depths, need = steps_needed(main)
    mu = np.nanmean(need, 0)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(depths, mu, color=SERIES[0], marker="o", ms=5, label="measured, choice accuracy ≥ 99%")
    d = np.array(depths, dtype=float)
    ax.plot(d, d, color=INK2, lw=1.2, ls=(0, (3, 2)), label="one hop per step")
    ok = ~np.isnan(mu) & (d >= 1)
    A = np.stack([np.log2(d[ok]), np.ones(ok.sum())], 1)
    coef = np.linalg.lstsq(A, mu[ok], rcond=None)[0]
    dl = d[d >= 1]
    ax.plot(dl, coef[0] * np.log2(dl) + coef[1], color=SERIES[1], lw=1.5,
            label=f"fit {coef[0]:.2f} log2 d {'+' if coef[1] >= 0 else '−'} {abs(coef[1]):.2f}")
    ax.axvspan(train_depth + 0.5, max(depths) + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
    ax.set_ylim(0, max(depths) + 1)
    ax.set_xlabel("chain depth")
    ax.set_ylabel("steps needed")
    ax.set_title("Steps needed grow sublinearly with depth")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_steps_needed.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--fixed", nargs="*", default=[])
    ap.add_argument("--untied", nargs="*", default=[])
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metric", default="choice", choices=["choice", "score", "noul", "all"])
    args = ap.parse_args()
    global METRIC
    METRIC = args.metric
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    td = load(args.main[0])[0]["train_depth"]
    fig_steps_curves(args.main, td, out)
    fig_heatmap(args.main, td, out)
    fig_compare(args.main, args.fixed, args.untied, td, out)
    fig_calibration(args.main, out)
    fig_auto(args.main, td, out)
    fig_steps_needed(args.main, td, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
