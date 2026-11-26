from src.evaluation.metrics import Evaluate
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.retrievers.base_retriever import Retriever
from src.data_utils.query import Query
import csv
from pathlib import Path

def evaluate_retriever(
    model: Retriever,
    queries: list[Query],
    k_values=(1, 5, 10, 100),
    batch_size=16
):
    top_k = max(k_values)
    relevancies = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]

        batch_relevancies = retrieveQueryAndGetRelevancies(
            model, batch, k=top_k
        )

        if any(len(row) != top_k for row in batch_relevancies):
            raise ValueError("Retriever returned fewer or more documents than requested")

        relevancies.extend(batch_relevancies)

    scores = Evaluate(relevancies, queries, k_values)

    return scores
def save_results(results, file_path):
    file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {file_path}")