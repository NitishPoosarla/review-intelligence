"""07-ai-review-intelligence — 04_insights.py — measured AI evaluation narrative."""
from pathlib import Path
import json
import sys

_shared_root = Path(__file__).resolve().parent
while _shared_root != _shared_root.parent and not (_shared_root / "shared").is_dir():
    _shared_root = _shared_root.parent
sys.path.insert(0, str(_shared_root))  # callable from repo root or standalone checkout
from shared import llm   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"


def main() -> None:
    ev = json.loads((OUT / "model_eval.json").read_text())
    ai = json.loads((OUT / "ai_eval.json").read_text())
    drivers = {
        "Baseline first": (f"TF-IDF holdout accuracy {ev['baseline_acc_holdout']:.1%} "
                           f"(F1 {ev['baseline_f1_holdout']:.1%}); majority floor "
                           f"{ev['majority_class_acc']:.1%} — the bar AI must clear."),
        "AI on the same rules": (f"AI classifier ({ai['mode']}) on a fixed sampled set: accuracy "
                                 f"{ai['ai_acc_sampled']:.1%}, F1 {ai['ai_f1_sampled']:.1%}, "
                                 f"latency {ai['latency_s_total']:.1f}s for {ai['chars_processed']:,} chars."),
        "Verdict": (f"beats baseline: {ai['ai_beats_baseline']} — an explicit, testable field rather "
                    "than a demo screenshot."),
    }
    prompt = f"""You are an AI/product analyst writing an evaluation memo for a CX leader deciding
whether to ship an LLM review-triage feature. Use ONLY the verified metrics below.
Deliver: (1) ship/no-ship recommendation with the deciding numbers, (2) evaluation plan for
scaled rollout, (3) honest risks (drift, ambiguity, cost).
Metrics: {json.dumps({**ev, **ai}, indent=1)}"""
    md = llm.write_insights(str(OUT / "insights.md"), "GenAI review-triage evaluation", ev, drivers, prompt)
    print(md[:900])


if __name__ == "__main__":
    main()
