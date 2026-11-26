import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

K_VALUES = [1, 5, 10, 100]


def load_results():
    results = []

    for file in sorted(RESULTS_DIR.glob("*_full.csv")):
        with file.open("r", encoding="utf-8") as f:
            results.extend(csv.DictReader(f))

    return results


def plot_metric(rows, dataset, metric):
    plt.figure(figsize=(8, 5))

    for row in rows:
        values = [
            float(row[f"{metric}@{k}"])
            for k in K_VALUES
        ]

        plt.plot(
            K_VALUES,
            values,
            marker="o",
            label=row["model"]
        )

    plt.xscale("log")
    plt.xticks(K_VALUES, K_VALUES)
    plt.xlabel("k")
    plt.ylabel(metric.capitalize())
    plt.ylim(bottom=0)
    plt.title(f"{dataset.upper()} - {metric.capitalize()}@k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{dataset}_{metric}.png",
        dpi=300
    )
    plt.savefig(
    FIGURES_DIR / f"{dataset}_{metric}.pdf",
    bbox_inches="tight"
)

    plt.close()


def plot_mrr(rows, dataset):
    models = [row["model"] for row in rows]
    values = [float(row["mrr"]) for row in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, width=0.4)
    plt.bar_label(bars, fmt="%.6f", padding=3)
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

    plt.ylabel("MRR@100")
    plt.title(f"{dataset.upper()} - MRR Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{dataset}_mrr.png",
        dpi=300
    )
    plt.savefig(
    FIGURES_DIR / f"{dataset}_mrr.pdf",
    bbox_inches="tight"
)

    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()

    for dataset, split in [("fiqa", "test"), ("squad", "val")]:

        rows = [
            row for row in results
            if row["dataset"].lower() == dataset
            and row["split"] == split
        ]

        if not rows:
            print(f"No results found for {dataset}")
            continue

        plot_metric(rows, dataset, "precision")
        plot_metric(rows, dataset, "recall")
        plot_mrr(rows, dataset)

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()