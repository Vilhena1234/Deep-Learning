from pathlib import Path
from datasets import load_dataset
from src.data_utils.query import Query
from src.retrievers.random_retriever import RandomRetriever
from src.evaluation.evaluate import evaluate_retriever, save_results


def main():

    # Load SQuAD validation dataset
    dataset = load_dataset(
        "rajpurkar/squad", split="validation"
    )

    # Extract unique passages
    unique_passages = list(dict.fromkeys(dataset["context"]))

    # Assign one ID to each unique passage
    passage_to_id = {
        text: f"doc{i}"
        for i, text in enumerate(unique_passages)
    }

    # Prepare document corpus
    documents = [
        {
            "text": [text],
            "document_id": [passage_to_id[text]]
        }
        for text in unique_passages
    ]

    # Prepare queries and relevant document IDs
    queries = [
        Query(
            text=row["question"],
            id=row["id"],
            relevant_document_ids=[
                passage_to_id[row["context"]]
            ]
        )
        for row in dataset
    ]

    print("Total queries:", len(queries))
    print("Total passages:", len(documents))

    # Initialize RandomRetriever
    model = RandomRetriever(
        documents=documents, seed=42
    )

    # Evaluate
    scores = evaluate_retriever(
        model, queries, k_values=[1, 5, 10, 100]
    )

    for metric, value in scores.items():
        print(f"{metric}: {value:.6f}")

    # Export results
    results = [{
        "dataset": "squad",
        "model": "RandomRetriever",
        "split": "val",
        "n_queries": len(queries),
        **scores
    }]

    save_results(
        results,
        Path(__file__).resolve().parent
        / "results/random_squad_full.csv"
    )


if __name__ == "__main__":
    main()