#!/usr/bin/env bash
# v2: the z-reason model of v1 on natural-language variable tracking (zreason/nl_tasks.py).
# Same configuration as run_experiments.sh; sequential on one RTX 3060, about 40 minutes per run.
set -euo pipefail
cd "$(dirname "$0")"
COMMON="--task nl --iters 36000 --bs 128 --lr 3e-4 --eval-every 2000"

train() {  # name, extra args; skipped when a trained checkpoint already exists
  [ -f "runs/$1/model.pt" ] && [ -f "runs/$1/done" ] && return
  python train.py --out "runs/$1" $COMMON "${@:2}" > "runs/$1.log" 2>&1
  touch "runs/$1/done"
}

train_pw() {  # ProofWriter: train on depth-3, character encoder, theories renamed so they cannot be memorized
  [ -f "runs/$1/model.pt" ] && [ -f "runs/$1/done" ] && return
  python train.py --out "runs/$1" --task proofwriter --features chars --max-chars 128 --pw-rename --train-depth 3 \
    --iters 24000 --bs 64 --lr 3e-4 --eval-every 2000 "${@:2}" > "runs/$1.log" 2>&1
  touch "runs/$1/done"
}

evaluate_pw() {  # deeper proofs (depth-5), human paraphrases (NatLang), another domain, in-distribution
  for pw in depth-5/test NatLang/test birds-electricity/test depth-3/test; do
    tag="_pw_${pw%%/*}"
    [ -f "runs/$1/eval$tag.json" ] || python eval.py "runs/$1" --task proofwriter --pw "$pw" --max-steps 64 \
      --tag "$tag" > "runs/$1.eval$tag.txt"
  done
}

evaluate() {  # depths 1-24 (3x the training depth), seen and held-out wording
  for split in train test; do
    [ -f "runs/$1/eval_nl_$split.json" ] || python eval.py "runs/$1" --task nl --split "$split" --depths 1-24 \
      --n 100 --max-steps 64 --tag "_nl_$split" > "runs/$1.eval_nl_$split.txt"
  done
}

# Runs with the first template set (4 training templates per kind) are archived in runs/nl_tmpl4:
# neither encoder generalized to held-out wording there. With the larger set, 12k iterations left
# the Laya-vector run undertrained (runs/nl_laya_12k_s0), hence 36k.
# ProofWriter: frozen Laya vectors (runs/pw_laya_memorized_s0) and plain characters memorized the
# 4.8k training theories; renaming each theory's vocabulary fixed it (runs/pw_rename_s0, 8k iterations).
evaluate_pw pw_rename_s0
train_pw pw_s0 --seed 0
evaluate_pw pw_s0
# one-pass baselines, same model and data. With the accuracy-gated curriculum they never get
# past short chains, so the level is forced to full difficulty by mid-training.
train_pw pw_fixed1_s0 --seed 0 --fixed-steps 1 --full-level-by 0.5
evaluate_pw pw_fixed1_s0
train nl_laya_s0 --seed 0 --features laya       # frozen Laya sentence vectors instead of characters
evaluate nl_laya_s0
train nl_laya_fixed1_s0 --seed 0 --features laya --fixed-steps 1 --full-level-by 0.5
evaluate nl_laya_fixed1_s0
train_pw pw_s1 --seed 1
evaluate_pw pw_s1
train_pw pw_s2 --seed 2
evaluate_pw pw_s2
train_pw pw_fixed8_s0 --seed 0 --fixed-steps 8
evaluate_pw pw_fixed8_s0
train_pw pw_untied8_s0 --seed 0 --untied 8      # transformer without weight tying
evaluate_pw pw_untied8_s0

# latency per step on an idle GPU (after all training)
[ -f runs/nl_laya_s0/latency.json ] || python -m analysis.latency runs/nl_laya_s0 --task nl > runs/nl_laya_s0.latency.txt 2>&1
[ -f runs/pw_s0/latency.json ] || python -m analysis.latency runs/pw_s0 --task proofwriter > runs/pw_s0.latency.txt 2>&1

train nl_laya_s1 --seed 1 --features laya
evaluate nl_laya_s1
train nl_laya_s2 --seed 2 --features laya
evaluate nl_laya_s2
