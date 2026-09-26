"""Jev-style prediction with a test-time `steps` knob."""
import json
import sys

from zreason.api import ZReason

m = ZReason.load(sys.argv[1] if len(sys.argv) > 1 else "runs/zr_s0")
state = "q=4;m=q;Z=m;c=7;k=c;w=2;T=w;A=T;B=A;C=B;D=C;E=D"
questions = {
    "value": {"type": "choice", "instructions": "val E", "criteria": {str(i): str(i) for i in range(10)}},
    "bucket": {"type": "score", "instructions": "lvl Z", "criteria": ["0-1", "2-3", "4-5", "6-7", "8-9"]},
    "depends": {"type": "noul", "instructions": "dep E w"},
}
for steps in (1, 2, 4, "auto"):
    print(f"--- steps={steps}")
    print(json.dumps(m.predict(state, questions, steps=steps), indent=1))
