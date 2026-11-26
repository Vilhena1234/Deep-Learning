import csv
from pathlib import Path
from plot_results import load_results

TABLES_DIR = Path(__file__).resolve().parent / "results" / "tables"

METRICS = {
    "precision@1": "P@1",
    "precision@5": "P@5",
    "precision@10": "P@10",
    "precision@100": "P@100",
    "recall@1": "R@1",
    "recall@5": "R@5",
    "recall@10": "R@10",
    "recall@100": "R@100",
    "mrr": "MRR@100"
}


def generate_table(results, dataset, split):

    rows = [
        row for row in results
        if row["dataset"].lower() == dataset
        and row["split"] == split
    ]

    if not rows:
        print(f"No results found for {dataset}")
        return

    table = []

    for row in rows:
        entry = {"Model": row["model"]}

        for metric, label in METRICS.items():
            entry[label] = f"{float(row[metric]):.6f}"

        table.append(entry)

    file_path = TABLES_DIR / f"{dataset}_comparison.csv"

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Model", *METRICS.values()]
        )
        writer.writeheader()
        writer.writerows(table)

    print(f"Table saved to {file_path}")
    latex_path = TABLES_DIR / f"{dataset}_comparison.tex"

    columns = ["Model", *METRICS.values()]

    with latex_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{tabular}{l" + "c" * len(METRICS) + "}\n")
        file.write("\\toprule\n")

        file.write(" & ".join(columns) + r" \\" + "\n")
        file.write("\\midrule\n")

        for row in table:
            values = [
                row[column].replace("_", r"\_")
                for column in columns
            ]

            file.write(" & ".join(values) + r" \\" + "\n")

        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")

    print(f"LaTeX table saved to {latex_path}")


def main():

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()

    generate_table(results, "fiqa", "test")
    generate_table(results, "squad", "val")


if __name__ == "__main__":
    main()