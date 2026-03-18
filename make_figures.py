from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parent
    train_csv = root / "training_data.csv"
    out_png = root / "fig_readmitted_distribution.png"

    df = pd.read_csv(train_csv)
    counts = df["readmitted"].value_counts(dropna=False)
    order = ["NO", ">30", "<30"]
    counts = counts.reindex([c for c in order if c in counts.index]).fillna(0).astype(int)

    plt.figure(figsize=(6.5, 4.0))
    bars = plt.bar(counts.index.astype(str), counts.values)
    plt.title("Readmission class distribution (training_data.csv)")
    plt.xlabel("readmitted class")
    plt.ylabel("count")

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

