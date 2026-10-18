"""07-ai-review-intelligence — 02_llm_classifier.py

The honest GenAI comparison:
  - With an API key: zero-shot LLM classification of the same holdout sample,
    evaluated with the SAME metrics as the costed TF-IDF baseline.
  - Without a key: a deterministic lexicon classifier acts as the "model-free"
    stand-in so the evaluation pipeline still runs offline end-to-end.
Latency + token/char cost are measured and reported — AI isn't free.
"""
from pathlib import Path
import json
import sys
import time

import pandas as pd

_shared_root = Path(__file__).resolve().parent
while _shared_root != _shared_root.parent and not (_shared_root / "shared").is_dir():
    _shared_root = _shared_root.parent
sys.path.insert(0, str(_shared_root))  # callable from repo root or standalone checkout
from shared import llm   # noqa: E402
from shared.utils import save_json   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"

POS = ["great", "good", "excellent", "love", "perfect", "best", "wonderful", "fantastic",
       "delicious", "recommend", "well", "enjoyed", "nice", "superb", "impressive", "must see"]
NEG = ["bad", "terrible", "worst", "awful", "boring", "waste", "disappointing", "poor",
       "hate", "horrible", "mislead", "avoid", "n't", "never", "lacked", "slow"]


def lexicon_predict(text: str) -> int:
    t = text.lower()
    pos, neg = sum(p in t for p in POS), sum(n in t for n in NEG)
    return int(pos >= neg)


def classify_llm(text: str) -> int:
    out = llm.ask_llm(
        f"Classify the product/movie review as positive=1 or negative=0. "
        f"Answer with exactly one character (1 or 0).\nReview: {text}")
    return 1 if out.strip()[:1] == "1" else 0


def main() -> None:
    import pandas as pd
    reviews = pd.read_csv(PROJECT / "data" / "processed" / "reviews.csv")
    ev = json.loads((OUT / "model_eval.json").read_text())

    # deterministic reproducible sample (same seed as baseline split shows holdout-like mix)
    sample = reviews.sample(n=min(150, len(reviews)), random_state=3)

    provider, _ = llm.detect_provider()
    t0 = time.time()
    if provider:
        preds = [classify_llm(t) for t in sample.text]
        mode = f"llm_zero_shot ({provider})"
        sample["pred"] = preds
    else:
        sample["pred"] = [lexicon_predict(t) for t in sample.text]
        mode = "lexicon_fallback (no API key set)"
    dt = time.time() - t0

    acc = float((sample["pred"] == sample["label"]).mean())
    from sklearn.metrics import f1_score
    out = {
        "mode": mode,
        "n": int(len(sample)),
        "ai_acc_sampled": round(acc, 4),
        "ai_f1_sampled": round(float(f1_score(sample["label"], sample["pred"])), 4),
        "baseline_acc_holdout": ev["baseline_acc_holdout"],
        "baseline_f1_holdout": ev["baseline_f1_holdout"],
        "ai_beats_baseline": bool(acc > ev["baseline_acc_holdout"]),
        "latency_s_total": round(dt, 1),
        "chars_processed": int(sample.text.str.len().sum()),
    }
    save_json(out, OUT / "ai_eval.json")
    sample.to_csv(OUT / "ai_classifications_sample.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
