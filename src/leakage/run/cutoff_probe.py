"""Empirical knowledge-cutoff probe.

Turns the *documented* model cutoffs into *measured* evidence: each model is asked about
events strictly before and strictly after its claimed cutoff. The treatment model should
answer 2024-H2 questions correctly (it was trained on that era); the control model should not;
neither should have specific knowledge of the 2026 OOD window. Results ground the experiment's
core assumption and are reported verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import CONTROL_MODEL, RESULTS_DIR, TREATMENT_MODEL  # noqa: E402

# (id, era, question, what a knowledgeable answer contains)
PROBES = [
    ("wc2022", "pre-2023 (both should know)",
     "Who won the 2022 FIFA World Cup? Answer in one sentence.", "Argentina"),
    ("election2024", "2024-H2 (target window)",
     "Who won the United States presidential election held in November 2024? One sentence.",
     "Trump"),
    ("aug2024crash", "2024-H2 (target window)",
     "On August 5, 2024 there was a sharp global stock-market selloff. What was the main "
     "cause? One sentence.", "yen carry trade unwind / Bank of Japan rate hike"),
    (" q1_2026", "2026 (OOD window — neither should know specifics)",
     "Describe one specific notable US stock-market event from the first quarter of 2026. "
     "One sentence. If you are not sure, say so.", "should express uncertainty"),
]


def _ask(model: str, question: str) -> str:
    import ollama
    try:
        r = ollama.chat(model=model, messages=[{"role": "user", "content": question}],
                        options={"temperature": 0.0, "num_predict": 200})
        return r["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"[error: {type(e).__name__}: {e}]"


def run(models=None) -> dict:
    models = models or [CONTROL_MODEL, TREATMENT_MODEL]
    out = {"models": {}}
    for spec in models:
        print(f"[cutoff-probe] {spec.tag} (documented cutoff: {spec.documented_cutoff})",
              flush=True)
        answers = []
        for pid, era, q, expect in PROBES:
            a = _ask(spec.tag, q)
            answers.append({"id": pid, "era": era, "question": q,
                            "expected_contains": expect, "answer": a})
            print(f"   [{pid}] {a[:160]}")
        out["models"][spec.tag] = {
            "documented_cutoff": spec.documented_cutoff,
            "label": spec.label,
            "answers": answers,
        }
    path = RESULTS_DIR / "cutoff_probe.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[cutoff-probe] wrote {path}")
    return out


if __name__ == "__main__":
    run()
