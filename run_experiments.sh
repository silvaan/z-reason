#!/usr/bin/env bash
# Reproduces every run, table and figure of the paper.
# Sequential on one GPU: about 48 minutes per run on an RTX 3060, about 4.5 hours in total.
set -euo pipefail
cd "$(dirname "$0")"
COMMON="--iters 12000 --bs 128 --lr 3e-4 --eval-every 2000"
MAIN="runs/zr_s0 runs/zr_s1 runs/zr_s2"

train() {  # name, extra args; skipped when a trained checkpoint already exists
  [ -f "runs/$1/model.pt" ] && [ -f "runs/$1/done" ] && return
  python train.py --out "runs/$1" $COMMON "${@:2}" > "runs/$1.log" 2>&1
  touch "runs/$1/done"
}

train zr_s0 --seed 0
train zr_s1 --seed 1
train zr_s2 --seed 2
train fixed8_s0 --seed 0 --fixed-steps 8      # same model, always trained with 8 steps
train untied8_s0 --seed 0 --untied 8          # transformer without weight tying, 8 two-block cores

for r in zr_s0 zr_s1 zr_s2 fixed8_s0 untied8_s0; do
  python eval.py "runs/$r" > "runs/$r.eval.txt"                                              # depths 1-16
  python eval.py "runs/$r" --depths 16-32 --n 250 --max-steps 64 --tag _deep > "runs/$r.eval_deep.txt"
done

python -m analysis.attention runs/zr_s0 --depth 16 --n 200 --steps 8
python -m analysis.trace_figure runs/zr_s0 --out figures/fig0_trace.png
python plots.py --main $MAIN --fixed runs/fixed8_s0 --untied runs/untied8_s0 --out figures
python paper/make_tables.py --main $MAIN --fixed runs/fixed8_s0 --untied runs/untied8_s0 --out paper/tables
cp figures/*.png paper/figures/
