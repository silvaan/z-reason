"""Which held-out wording breaks generalization? Hold out one template group at a time.

Facts and questions use the training templates except for one group (root facts, copy facts
or questions), which uses only its held-out templates. Depths 1-8, accuracy over all
question types at 1, 4 and 16 steps.

    python -m analysis.heldout_ablation runs/nl_laya_s0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import zreason.nl_tasks as nt
from eval import run_eval
from train import nl_lines
from zreason.api import ZReason

GROUPS = {"none": [], "root": ["ROOT"], "copy": ["COPY"], "question": ["CHOICE_Q", "SCORE_Q", "NOUL_Q"],
          "all": ["ROOT", "COPY", "CHOICE_Q", "SCORE_Q", "NOUL_Q"]}
STEPS = (1, 4, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    api = ZReason.load(args.run)
    orig = nt.templates
    res = {}
    for name, held in GROUPS.items():
        nt.templates = lambda g, split, held=held: orig(g, "test" if g in held else "train")
        rng = random.Random(0)
        exs = [nl_lines(nt.make_nl_example(rng, d, n_questions=4)) for d in (1, 2, 4, 8) for _ in range(args.n)]
        rows, _ = run_eval(api, exs, max(STEPS))
        r = np.array([(s, h) for _, _, s, h, _ in rows])
        res[name] = {s: float(r[r[:, 0] == s, 1].mean()) for s in STEPS}
        print(name, {s: round(a, 3) for s, a in res[name].items()}, flush=True)
    nt.templates = orig
    (Path(args.run) / "heldout_ablation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
