"""07-ai-review-intelligence — 03_dashboard.py — model evaluation dashboard."""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

_shared_root = Path(__file__).resolve().parent
while _shared_root != _shared_root.parent and not (_shared_root / "shared").is_dir():
    _shared_root = _shared_root.parent
sys.path.insert(0, str(_shared_root))  # callable from repo root or standalone checkout
from shared.dashboard import build_dashboard   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"


def main() -> None:
    ev = json.loads((OUT / "model_eval.json").read_text())
    ai = json.loads((OUT / "ai_eval.json").read_text())
    df = pd.read_csv(PROJECT / "data" / "processed" / "reviews.csv")

    byc = pd.DataFrame(ev["by_corpus"]).T.reset_index(names="corpus")
    fig_bc = go.Figure(go.Bar(x=byc.corpus, y=byc.acc * 100, marker_color="#134e5e",
                              text=[f"{a:.0%}" for a in byc.acc], textposition="outside"))
    fig_bc.update_layout(yaxis_title="Holdout accuracy (TF-IDF baseline)", xaxis_title="")

    fig_cmp = go.Figure(go.Bar(
        x=["majority class", "TF-IDF baseline", "AI classifier (sampled)"],
        y=[ev["majority_class_acc"] * 100, ev["baseline_acc_holdout"] * 100, ai["ai_acc_sampled"] * 100],
        marker_color=["#94a3b8", "#134e5e", "#e2584c"],
        text=[f"{v:.1%}" for v in [ev["majority_class_acc"], ev["baseline_acc_holdout"], ai["ai_acc_sampled"]]],
        textposition="outside"))
    fig_cmp.update_layout(yaxis_title="Accuracy")

    # word count split by label
    df["wc"] = df.text.str.split().str.len()
    g = df.groupby("label").wc.mean()
    fig_wc = px_bar = go.Figure(go.Bar(x=[f"negative", "positive"], y=[g.get(0, 0), g.get(1, 0)],
                                       marker_color="#0e7490"))
    fig_wc.update_layout(yaxis_title="Avg words per review", xaxis_title="")

    build_dashboard(
        str(PROJECT / "dashboard" / "index.html"),
        title="GenAI Review Intelligence — With the Evaluation RECEIPT",
        subtitle=("Can AI triage product reviews, and how do we PROVE its accuracy against a costed "
                  "baseline? 3,000 labelled reviews (Amazon/IMDB/Yelp, public); same-metric comparison."),
        badges=["AI/GenAI", "Model evaluation", "Latency & cost", "LLM z\u00e9ro-shot"],
        kpis=[
            {"label": "Labelled reviews", "value": f"{ev['n_examples']:,}"},
            {"label": "TF-IDF baseline acc", "value": f"{ev['baseline_acc_holdout']:.1%}", "note": "25% holdout"},
            {"label": "AI classifier acc", "value": f"{ai['ai_acc_sampled']:.1%}", "note": f"sampled n={ai['n']}, {ai['mode'].split(' (')[0]}"},
            {"label": "AI beats baseline", "value": ("yes" if ai["ai_beats_baseline"] else "no"), "note": "same metrics"},
            {"label": "Latency", "value": f"{ai['latency_s_total']:.0f}s", "note": f"{ai['chars_processed']:,} chars"},
        ],
        figures=[
            {"title": "Accuracy comparison", "sub": "majority class versus costed baseline versus AI — same dataset rules", "fig": fig_cmp},
            {"title": "Baseline accuracy by corpus", "fig": fig_bc},
            {"title": "Review length by sentiment", "fig": fig_wc},
        ],
        recommendations=[
            "Ship the cheap baseline first: it is strong, instant, costs nothing at inference — AI only where it adds lift.",
            "Evaluate any LLM classifier with the SAME holdout metrics and report latency + cost per 1k reviews.",
            "Sample-and-audit: keep a 150-row human-audited sample in the loop (ai_classifications_sample.csv).",
            "Route ambiguous predictions (p between .45-.55) to human review instead of forcing 0/1.",
        ],
        footer=("Data: UCI Sentiment Labelled Sentences (via public Microsoft mirror). AI narrates verified "
                "metrics; its own classification is itself an *evaluated artifact*, not a assumption."),
    )
    print("dashboard written")


if __name__ == "__main__":
    main()
