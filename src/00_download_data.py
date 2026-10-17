"""07-ai-review-intelligence — download 3 labelled review corpora (public).

Source mirrors: microsoft/ML-Server-Python-Samples copies of the UCI
"Sentiment Labelled Sentences" files (Amazon cells, IMDB, Yelp; 1,000 sentences each,
0/1 ground-truth labels — the zero-shot evaluation gold set).
"""
from pathlib import Path
import sys

_shared_root = Path(__file__).resolve().parent
while _shared_root != _shared_root.parent and not (_shared_root / "shared").is_dir():
    _shared_root = _shared_root.parent
sys.path.insert(0, str(_shared_root))  # callable from repo root or standalone checkout
from shared.utils import download   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
BASE = ("https://raw.githubusercontent.com/microsoft/ML-Server-Python-Samples/"
        "master/microsoftml/202/data/sentiment_analysis/")
FILES = ["amazon_cells_labelled", "imdb_labelled", "yelp_labelled"]


def main() -> None:
    for f in FILES:
        download(BASE + f + ".txt", PROJECT / "data" / "raw" / f"{f}.txt", f + ".txt")


if __name__ == "__main__":
    main()
