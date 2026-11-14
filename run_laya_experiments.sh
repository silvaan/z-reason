#!/usr/bin/env bash
# z-Laya runs for the v2 preprint: z-reason on top of Laya's frozen encoder, on the
# natural-language variable tracking task. Sequential on one RTX 3060.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0

[ -f cache/nl_train/index.pt ] || python cache_laya.py --out cache/nl_train --n-states 30000 --seed 0 > cache/nl_train.log 2>&1

train() {  # name, extra args; skipped when a trained head already exists
  [ -f "runs/$1/head.pt" ] && [ -f "runs/$1/done" ] && return
  python train_laya.py --out "runs/$1" "${@:2}" > "runs/$1.log" 2>&1
  touch "runs/$1/done"
}

evaluate() {
  for split in train test; do
    [ -f "runs/$1/eval_$split.json" ] || python eval_laya.py "runs/$1" --split "$split" "${@:2}" >> "runs/$1.eval.txt" 2>&1
  done
}

evaluate laya_zeroshot --zero-shot                 # Laya as released, one pass
train zl_s0 --seed 0                               # random steps
evaluate zl_s0
train zl_fixed1_s0 --seed 0 --fixed-steps 1        # one pass, fine-tuned: the System-One baseline
evaluate zl_fixed1_s0
train zl_untied8_s0 --seed 0 --untied 8            # 8 distinct heads: same compute as 8 steps, 8x the head parameters
evaluate zl_untied8_s0
train zl_s1 --seed 1
evaluate zl_s1
train zl_s2 --seed 2
evaluate zl_s2
train zl_tokens_s0 --seed 0 --units tokens         # recurrence over tokens instead of facts: the v1 failure, at scale
evaluate zl_tokens_s0
