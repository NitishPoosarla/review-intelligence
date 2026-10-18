"""07-ai-review-intelligence — 01_baseline_model.py

Train/test the *costed baseline* a GenAI classifier must beat:
TF-IDF + LogisticRegression, per-corpus and pooled, with holdout accuracy/F1.
"""
from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

_shared_root = Path(__file__).resolve().parent
while _shared_root != _shared_root.parent and not (_shared_root / "shared").is_dir():
    _shared_root = _shared_root.parent
sys.path.insert(0, str(_shared_root))  # callable from repo root or standalone checkout
from shared.utils import save_json, save_png   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
mpl_needed = True
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "semibold"})
OUT = PROJECT / "outputs"


def main() -> None:
    (PROJECT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for src in ["amazon_cells", "imdb", "yelp"]:
        d = pd.read_csv(PROJECT / "data" / "raw" / f"{src}_labelled.txt", sep="\t", header=None,
                        names=["text", "label"])
        d["corpus"] = src
        frames.append(d)
    df = pd.concat(frames).dropna().drop_duplicates(subset=["text"])
    df.to_csv(PROJECT / "data" / "processed" / "reviews.csv", index=False)

    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"],
                                              test_size=.25, stratify=df["label"], random_state=3)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    evals = {
        "n_examples": int(len(df)),
        "baseline_acc_holdout": round(float(accuracy_score(y_te, pred)), 4),
        "baseline_f1_holdout": round(float(f1_score(y_te, pred)), 4),
        "majority_class_acc": round(float(max(y_tr.mean(), 1 - y_tr.mean())), 4),
        "by_corpus": {},
    }
    for src in ["amazon_cells", "imdb", "yelp"]:
        m = df.corpus == src
        Xi, yi = df[m].text, df[m].label
        a, b, c, e = train_test_split(Xi, yi, test_size=.25, stratify=yi, random_state=3)
        pipe.fit(a, c)
        evals["by_corpus"][src] = {
            "acc": round(float(accuracy_score(e, pipe.predict(b))), 4),
            "f1": round(float(f1_score(e, pipe.predict(b))), 4),
        }
    save_json(evals, OUT / "model_eval.json")

    # confusion by corpus (holdout pooled)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["neg", "pos"]); ax.set_yticks([0, 1], ["neg", "pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_title("TF-IDF baseline confusion (holdout)")
    save_png(fig, OUT / "charts" / "confusion.png")
    print(json.dumps(evals, indent=2))


if __name__ == "__main__":
    main()
