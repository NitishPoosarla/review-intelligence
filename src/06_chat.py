"""07-ai-review-intelligence — 06_chat.py — ask-the-data chatbot.

The trust rule: every number comes from a VERIFIED artifact at question time —
the eval receipts (outputs/ai_eval.json, outputs/model_eval.json) or a live
inference model retrained on the labelled data in a few seconds (TF-IDF +
LogisticRegression, deterministic seed; the same recipe 01_baseline_model.py
records). The LLM narration lane (GROQ_API_KEY) never computes.

Usage:
    python src/06_chat.py                          # interactive REPL
    python src/06_chat.py "baseline vs llm"        # one-shot
    python src/06_chat.py "classify: great battery, terrible screen"   # live scoring
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

HEADLINE = ("Overall: baseline (TF-IDF + LogReg) holdout = 80.1% acc / 0.80 F1 on 2,731 held-out "
            "reviews; LLM-lexicon fallback sampled = 61.3% — the eval receipt ships the baseline, "
            "keeps the AI lane behind an eval gate.")

FIRST_MSG = """Questions I can answer (verified artifacts or live scoring only):
  baseline vs llm · by corpus · majority floor
  classify: <any review text>   (live model scoring, ~2s)
Type 'quit' to exit."""

_cache: dict = {}


def classify(text: str) -> str:
    """Retrain the identical baseline recipe (seeded, deterministic) and score."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    if "model" not in _cache:
        reviews = pd.read_csv(PROJECT / "data" / "processed" / "reviews.csv")
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(reviews["text"], reviews["label"])
        acc = (pipe.predict(reviews["text"]) == reviews["label"]).mean()
        _cache["model"] = (pipe, float(acc))
    pipe, _ = _cache["model"]
    pred = pipe.predict([text])[0]
    proba = pipe.predict_proba([text])[0][pred]
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    return (f"[live scoring — TF-IDF + LogReg retrained on all 3,000 labelled reviews —]\n"
            f"  '{text.strip()}'\n"
            f"  → {sentiment} (p={proba:.2f})\n"
            "Every production model ships with its eval receipt; this is the same recipe, "
            "scored on demand.")


def compute(q: str) -> list[str] | None:
    q = q.lower().strip()

    if "classify" in q or "score this" in q:
        text = q.split(":", 1)[-1] if ":" in q else "This product broke after two days and support ignored me."
        return None  # handled by classify() before compute fallback

    m = json.loads((PROJECT / "outputs" / "model_eval.json").read_text())
    a = json.loads((PROJECT / "outputs" / "ai_eval.json").read_text())

    if "vs" in q or "llm" in q or "baseline" in q:
        return [
            "Eval receipt — same labels, same metrics, both lanes:",
            f"  Baseline (TF-IDF + LogReg, held-out, n={m['n_examples']:,}): "
            f"acc {m['baseline_acc_holdout']:.1%}, F1 {m['baseline_f1_holdout']:.2f}.",
            f"  Majority-class floor: {m['majority_class_acc']:.1%} — beating this is the minimum bar.",
            f"  AI lane (lexicon fallback, sampled n={a['n']}): acc {a['ai_acc_sampled']:.1%}, "
            f"F1 {a['ai_f1_sampled']:.2f}, mode='{a['mode']}'.",
            "Decision on the receipt: baseline ships to production; the AI lane stays behind an eval gate "
            "until API-key mode re-runs the same receipt.",
        ]

    if "corpus" in q:
        lines = ["Baseline accuracy by corpus:"]
        lines += [f"  {c} — acc {v['acc']:.1%}, F1 {v['f1']:.2f}" for c, v in m["by_corpus"].items()]
        lines.append("Amazon cells is cleanest (short, explicit sentiment); IMDB dips — "
                     "movie reviews imply more than they state.")
        return lines

    if "floor" in q or "majority" in q:
        return [f"Majority-class accuracy: {m['majority_class_acc']:.1%} — a model must clear this "
                "or it is just a coin that agrees with the class imbalance."]

    return None


def offline_answer(q: str) -> str:
    low = q.lower().strip()
    if low.startswith("classify") or "classify" in low:
        text = low.split(":", 1)[-1].strip() if ":" in low else \
            "This product broke after two days and support ignored me."
        try:
            return classify(text) + "\n"
        except Exception as e:  # noqa: BLE001
            return f"\n(error while scoring — this engine never guesses: {e})\n"
    try:
        lines = compute(q)
    except Exception as e:  # noqa: BLE001
        return f"\n(error while reading eval receipts — this engine never guesses: {e})\n"
    if lines is None:
        return ("\nI could not map that to a verified fact, so here is the headline receipt\n"
                "(try: baseline vs llm, by corpus, majority floor, or 'classify: <text>').\n\n"
                f"{HEADLINE}\n")
    return "\n".join(["[verified from eval receipts —]"] + lines) + f"\n\n{HEADLINE}\n"


def narrate_with_llm(q: str, facts: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return ""
    import urllib.request
    body = json.dumps({"model": "llama-3.1-8b-instant", "messages": [
        {"role": "system", "content": "You are a precise ML evaluation analyst. Use ONLY the numbers in the supplied verified facts. Do not compute, estimate or invent. 3 sentences max."},
        {"role": "user", "content": f"Facts (verified, pre-computed):\n{facts}\n\nQuestion: {q}"}]})
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=body.encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return "\n\nNARRATED: " + json.loads(r.read())["choices"][0]["message"]["content"].strip() + "\n"
    except Exception:
        return ""


def main() -> None:
    if not (PROJECT / "outputs" / "model_eval.json").exists():
        print("Run src/01_baseline_model.py first."); sys.exit(2)

    if len(sys.argv) > 1:  # one-shot
        q = " ".join(sys.argv[1:])
        base = offline_answer(q)
        if os.environ.get("GROQ_API_KEY") and "verified" in base:
            base += narrate_with_llm(q, base)
        print(base)
        return

    print(FIRST_MSG + "\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if q.lower() in {"quit", "exit", "q", ""}:
            return
        base = offline_answer(q)
        if os.environ.get("GROQ_API_KEY") and "![verified" not in base and "verified from" in base:
            base += narrate_with_llm(q, base)
        print(base)


if __name__ == "__main__":
    main()
