"""07 — banner: paper scorecard, baseline vs AI lane. No decorative gradient."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

m = json.loads((Path(__file__).resolve().parents[1] / "outputs" / "model_eval.json").read_text())
a = json.loads((Path(__file__).resolve().parents[1] / "outputs" / "ai_eval.json").read_text())

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
ax.set_xlim(0, 12); ax.set_ylim(0, 6.75); ax.axis("off")
fig.patch.set_facecolor("#f4f2ee")

ax.text(0.5, 6.05, "THE RECEIPTS SHEET: DOES THE AI LANE EARN ITS KEEP?",
        fontsize=17, fontweight="bold", color="#20242a")
ax.text(0.5, 5.5, "same labels, same sample discipline — 3,000 reviews (UCI labelled sentences)", fontsize=10, color="#6b7280")

boxes = [
    ("BASELINE\nTF-IDF + Logistic Reg.",
     f"{m['baseline_acc_holdout']:.1%}",
     "F1 " + f"{m['baseline_f1_holdout']:.2f}", 0.5),
    ("AI LANE\nLLM classifier, zero-shot", f"{a['ai_acc_sampled']:.1%}",
     "F1 " + f"{a['ai_f1_sampled']:.2f}", 6.1),
]
for title, acc, f1, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 2.4), 5.4, 2.6, boxstyle="round,pad=0.08",
                                fc="white", ec="#c9c5bc"))
    ax.text(x + 2.7, 4.55, title, fontsize=11.5, color="#6b7280", ha="center",
            va="top", linespacing=1.4)
    ax.text(x + 1.7, 3.15, acc, fontsize=30, fontweight="bold", color="#20242a",
            ha="center")
    ax.text(x + 2.7, 2.62, f1, fontsize=11, color="#6b7280", ha="center")

verdict = "AI WINS" if a.get("ai_beats_baseline") else "THE BASELINE SHIPS"
ax.add_patch(FancyBboxPatch((3.7, 0.75), 4.6, 0.85, boxstyle="round,pad=0.06",
                            fc="#20242a", ec="none"))
ax.text(6.0, 1.18, verdict, fontsize=14, fontweight="bold", color="white", ha="center")
ax.text(6.0, 0.28, f"majority-class floor shatters at {m['majority_class_acc']:.1%} — everything above it was earned",
        fontsize=9.5, color="#6b7280", ha="center")
fig.savefig(Path(__file__).resolve().parents[1] / "outputs" / "banner.png",
            dpi=100, facecolor=fig.get_facecolor())
print("banner written")
