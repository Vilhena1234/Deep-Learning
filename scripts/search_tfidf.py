from data import load
from src.retrievers.tf_idf import TFIDF


def main():

    print("=" * 60)
    print("             TF-IDF PASSAGE SEARCH")
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

    # Load the complete corpus
    print(f"\nLoading {dataset.upper()} dataset...")

    corpus, queries, qrels = load(dataset, split)

    print("Total passages:", len(corpus))

    # Convert to the format expected by DocumentDataset
    documents = [{"text": list(corpus.values()), "document_id": list(corpus.keys())}]

    # Build TF-IDF index
    print("\nBuilding TF-IDF index. Please wait...")

    retriever = TFIDF(documents)

    print("\nTF-IDF index successfully built!")
    print("Documents indexed:", len(retriever.index.GetDocuments()))
    print("Vocabulary size:", len(retriever.corpus_vocabulary))

    # Interactive search
    while True:

        query = input("\nEnter search query (or type 'exit'): ").strip()

        if query.lower() in ["exit", "quit"]:
            print("\nSearch session ended.")
            break

        if not query:
            print("Please enter a search query.")
            continue

        # Calculate TF-IDF scores
        scores = retriever.CalculateScores([query])[0]

        all_documents = retriever.index.GetDocuments()

        # Rank by score, highest first
        ranked_results = sorted(
            zip(scores, all_documents), key=lambda item: float(item[0]), reverse=True
        )

        # Display the five highest-scoring passages
        results = [
            (score, document) for score, document in ranked_results if float(score) > 0
        ][:5]

        print("\n" + "=" * 60)
        print("SEARCH RESULTS")
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
