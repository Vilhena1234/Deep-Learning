from data import load
from src.retrievers.bm25 import BM25


def main():

    print("=" * 60)
    print("              BM25 PASSAGE SEARCH")
    print("=" * 60)

    # Select dataset
    print("\nChoose your dataset:")
    print("1. FiQA")
    print("2. SQuAD")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        dataset = "fiqa"
        split = "test"

    elif choice == "2":
        dataset = "squad"
        split = "dev"

    else:
        print("Invalid choice.")
        return

    # Load complete dataset
    print(f"\nLoading {dataset.upper()} dataset...")

    corpus, queries, qrels = load(dataset, split)

    print("Total passages:", len(corpus))

    # Convert corpus into DocumentDataset format
    documents = [{"text": list(corpus.values()), "document_id": list(corpus.keys())}]

    # Build BM25 index
    print("\nBuilding BM25 index. Please wait...")

    retriever = BM25(documents, k1=1.2, b=0.75)

    print("\nBM25 index successfully built!")
    print("Documents indexed:", len(retriever.index.GetDocuments()))
    print("Vocabulary size:", len(retriever.corpus_vocabulary))
    print("Average document length:", retriever.average_document_length)

    # Interactive search loop
    while True:

        query = input("\nEnter search query (or type 'exit'): ").strip()

        if query.lower() in ["exit", "quit"]:
            print("\nSearch session ended.")
            break

        if not query:
            print("Please enter a search query.")
            continue

        # Calculate BM25 scores
        scores = retriever.CalculateScores([query])[0]

        all_documents = retriever.index.GetDocuments()

        # Rank documents by score
        ranked_results = sorted(
            zip(scores, all_documents), key=lambda item: float(item[0]), reverse=True
        )

        # Get top five matching passages
        results = [
            (score, document) for score, document in ranked_results if float(score) > 0
        ][:5]

        print("\n" + "=" * 60)
        print("BM25 SEARCH RESULTS")
        print("=" * 60)
        print("Query:", query)

        if not results:
            print("\nNo matching passages found.")
            print("Try different keywords.")
            continue

        for rank, (score, document) in enumerate(results, 1):

            print(f"\nRank: {rank}")
            print(f"Score: {float(score):.4f}")
            print(f"Document ID: {document.GetId()}")
            print("\nPassage:")
            print(document.GetText())
            print("-" * 60)


if __name__ == "__main__":
    main()
