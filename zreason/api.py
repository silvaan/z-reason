"""Jev-compatible prediction API with a test-time `steps` knob."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .batching import collate, option_names, to_device
from .model import ZConfig, ZReasoner
from .tokenizer import CharTokenizer


def confidence(p: list[float]) -> float:
    """1 - normalised entropy: 1.0 when all mass sits on one option, 0.0 when uniform."""
    k = len(p)
    if k < 2:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(k))


def format_answer(q: dict, logits: torch.Tensor, noul_logit: torch.Tensor, temperature: float = 1.0) -> dict:
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(torch.sigmoid(noul_logit / temperature).item(), 4)}
    names = option_names(q)
    p = torch.softmax(logits[: len(names)].float() / temperature, -1).tolist()
    probs = {n: round(x, 4) for n, x in zip(names, p)}
    ans = {"type": q["type"]}
    if q["type"] == "choice":
        ans["choice"] = names[max(range(len(p)), key=p.__getitem__)]
    else:
        crit = q["criteria"]
        texts = list(crit.values()) if isinstance(crit, dict) else list(crit)
        ans["score"] = round(sum(i * x for i, x in enumerate(p)), 4)
        ans["legend"] = {str(i): t for i, t in enumerate(texts)}
    ans["confidence"] = round(confidence(p), 4)
    ans["probabilities"] = probs
    return ans


class ZReason:
    def __init__(self, model: ZReasoner, device="cuda", name="zreason-0.1"):
        self.model = model.to(device).eval()
        self.device = device
        self.tok = CharTokenizer()
        self.name = name

    @classmethod
    def load(cls, run_dir: str | Path, device="cuda"):
        run_dir = Path(run_dir)
        cfg = ZConfig(**json.loads((run_dir / "config.json").read_text())["model"])
        model = ZReasoner(cfg)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
        return cls(model, device=device, name=f"zreason-{run_dir.name}")

    @torch.no_grad()
    def predict(self, state, questions: dict, steps: int | str = 16, max_steps: int = 128,
                tol: float = 1e-3, seed: int | None = 0, trace: bool = False) -> dict:
        """steps: int, or "auto" = iterate until answer distributions stop moving
        (max total-variation change < tol) or `max_steps` is reached."""
        batch = to_device(collate([{"state": state, "questions": questions}], self.tok), self.device)
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        auto = steps == "auto"
        n = max_steps if auto else int(steps)
        out = self.model(batch, steps=n, trace=trace or auto, generator=g)

        used = n
        if auto:
            prev = None
            for i, o in enumerate(out["trace"]):
                cur = torch.cat([o["logits"].float().softmax(-1).nan_to_num(0), torch.sigmoid(o["noul_logit"]).unsqueeze(-1)], -1)
                if prev is not None and (cur - prev).abs().max().item() < tol:
                    used, out = i + 1, dict(o, trace=out["trace"][: i + 1])
                    break
                prev = cur

        answers = {}
        qids = [qid for _, qid in batch["q_ids"]]
        for i, qid in enumerate(qids):
            answers[qid] = format_answer(questions[qid], out["logits"][i], out["noul_logit"][i])
        res = {
            "model": self.name,
            "answers": answers,
            "steps": used,
            "usage": {"input_tokens": batch["n_tokens"], "output_tokens": 0},
        }
        if trace:
            res["trace"] = [
                {qid: format_answer(questions[qid], o["logits"][i], o["noul_logit"][i]) for i, qid in enumerate(qids)}
                for o in out["trace"][:used]
            ]
        return res
