"""Empirically select the treatment model.

The experiment's premise requires a treatment model that *demonstrably* knows the target
window (2024-H2) and a control that does not. Self-reported cutoffs are unreliable (qwen3:8b
claims "October 2023" yet is a 2025 release), so we *measure* it: probe each candidate with
specific, hard-to-confabulate 2024-H2 facts and score keyword recall. The highest-scoring model
that still denies knowledge of the 2026 OOD window is the valid treatment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RESULTS_DIR  # noqa: E402

# (id, question, list-of-acceptable-keywords any-of)  — specific 2024-H2 facts.
DISCRIMINATORS = [
    ("election2024",
     "Who won the November 2024 US presidential election? Name the president-elect and the "
     "vice president-elect.", [["trump"], ["vance"]]),
    ("harris_vp",
     "In August 2024, who did Kamala Harris pick as her vice-presidential running mate?",
     [["walz"]]),
    ("nvda_split",
     "In June 2024 a major AI chip company did a 10-for-1 stock split. Which company?",
     [["nvidia", "nvda"]]),
    ("yen_carry",
     "What triggered the sharp global stock selloff around August 5, 2024 that started in "
     "Japan?", [["carry trade", "yen carry", "bank of japan", "boj rate"]]),
]
SANITY = ("wc2022", "Who won the 2022 FIFA World Cup?", [["argentina"]])
OOD_DENY = ("q1_2026",
            "Name one specific US stock-market event from Q1 2026. If unsure, say you don't know.",
            None)


def _ask(model: str, q: str) -> str:
    import ollama
    think = any(k in model.lower() for k in ("qwen3", "deepseek-r1", "r1"))
    kw = dict(model=model, messages=[{"role": "user", "content": q}],
              options={"temperature": 0.0, "num_predict": 250})
    try:
        try:
            r = ollama.chat(**({**kw, "think": False} if think else kw))
        except Exception:  # noqa: BLE001
            r = ollama.chat(**kw)
        return r["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"


def _scores(answer: str, groups) -> int:
    """1 if the answer satisfies ALL keyword-groups (each group = any-of), else 0."""
    a = answer.lower()
    return int(all(any(k in a for k in grp) for grp in groups))


def probe(models: list[str]) -> dict:
    out = {}
    for m in models:
        print(f"\n### {m}", flush=True)
        rec = {"discriminators": {}, "score_2024H2": 0}
        for pid, q, groups in DISCRIMINATORS:
            ans = _ask(m, q)
            s = _scores(ans, groups)
            rec["discriminators"][pid] = {"score": s, "answer": ans}
            rec["score_2024H2"] += s
            print(f"  [{pid}] {'✓' if s else '✗'} {ans[:130]}")
        for pid, q, groups in (SANITY,):
            ans = _ask(m, q)
            rec[pid] = {"score": _scores(ans, groups), "answer": ans}
            print(f"  [{pid}] {'✓' if rec[pid]['score'] else '✗'} {ans[:90]}")
        ans = _ask(m, OOD_DENY[1])
        rec["q1_2026_answer"] = ans
        # A valid treatment must DENY specific 2026 knowledge (else C-B is not truly OOD for it).
        al = ans.lower()
        rec["denies_2026"] = (any(c in al for c in (
            "don't know", "do not know", "not aware", "unable", "cannot", "can't",
            "no specific", "i don't have", "as of my", "unsure", "not sure",
            "no information", "haven't")) and not ans.startswith("[error"))
        rec["sanity_ok"] = bool(rec.get(SANITY[0], {}).get("score", 0))
        print(f"  [q1_2026 denies={rec['denies_2026']}] {ans[:120]}")
        rec["max_2024H2"] = len(DISCRIMINATORS)
        out[m] = rec
    path = RESULTS_DIR / "model_selection.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[model-selection] wrote {path}")
    print("\nSCOREBOARD (2024-H2 factual recall):")
    for m, r in sorted(out.items(), key=lambda kv: -kv[1]["score_2024H2"]):
        print(f"  {m:<16} {r['score_2024H2']}/{r['max_2024H2']}")
    return out


def pick_best(models: list[str], min_score: int = 2) -> str:
    """Select the treatment model: highest 2024-H2 recall among candidates that ALSO deny 2026
    knowledge and pass the sanity probe, with score >= min_score. Fails loudly otherwise so a
    broken/non-functional model can never be silently promoted to the treatment slot.
    """
    res = probe(models)
    eligible = [m for m in models
                if res[m].get("denies_2026") and res[m].get("sanity_ok")
                and res[m]["score_2024H2"] >= min_score]
    if not eligible:
        raise RuntimeError(
            "No candidate qualifies as treatment (need score_2024H2>="
            f"{min_score}, denies_2026, sanity_ok). Scores: "
            + ", ".join(f"{m}={res[m]['score_2024H2']}/4 "
                        f"deny={res[m].get('denies_2026')} sane={res[m].get('sanity_ok')}"
                        for m in models))
    best = max(eligible, key=lambda m: res[m]["score_2024H2"])
    (RESULTS_DIR / "treatment_model.txt").write_text(best)
    return best


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pick":
        # Probe the candidates, write/emit only the winning tag (stdout last line) for scripts.
        winner = pick_best(sys.argv[2:] or ["qwen3:32b", "qwen2.5:32b"])
        print(winner)
    else:
        probe(sys.argv[1:] or ["llama3.1:8b", "qwen3:8b", "phi4", "qwen2.5:7b"])
