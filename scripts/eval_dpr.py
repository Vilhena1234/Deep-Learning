import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.retrievers.dense import DenseRetriever  # noqa: E402

KS = (1, 5, 10, 100)


def evaluate(retriever, queries, qrels):
    documents = retriever.index.GetDocuments()
    question_ids = list(queries)
    results = retriever.TopK([queries[q] for q in question_ids], k=max(KS))
    totals = {f"P@{k}": 0.0 for k in KS} | {f"R@{k}": 0.0 for k in KS} | {"MRR": 0.0}
    for q, ranked in zip(question_ids, results):
        relevant = {d for d, value in qrels[q].items() if value > 0}
        hits = [documents[j].GetId() in relevant for j, _ in ranked]
        for k in KS:
            totals[f"P@{k}"] += sum(hits[:k]) / k
            totals[f"R@{k}"] += sum(hits[:k]) / len(relevant)
        if True in hits:
            totals["MRR"] += 1 / (hits.index(True) + 1)
    return {name: round(value / len(question_ids), 4) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name or checkpoint folder")
    parser.add_argument("--name", required=True, help="row name in the CSV")
    parser.add_argument("--limit", type=int, help="use only this many questions (for testing)")
    parser.add_argument("--out", help="CSV file to add the row to")
    args = parser.parse_args()

    with (ROOT / "data" / "processed" / "squad_dev.json").open(encoding="utf-8") as file:
        dataset = json.load(file)
    ids = list(dataset["qrels"])
    if args.limit is not None:
        random.Random(37).shuffle(ids)
        ids = ids[:args.limit]
    queries = {q: dataset["queries"][q] for q in ids}

    corpus = dataset["corpus"]
    documents = [{"text": list(corpus.values()), "document_id": list(corpus.keys())}]
    retriever = DenseRetriever(documents, args.model, vectors_path=ROOT / "index" / f"{args.name}.pt")

    row = {"method": args.name, "questions": len(queries), "passages": len(corpus)}
    row |= evaluate(retriever, queries, dataset["qrels"])
    print(row)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out.exists()
        with out.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
