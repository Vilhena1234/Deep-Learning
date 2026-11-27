from data import load
from src.retrievers.bm25 import BM25

datasets = [("FiQA", "fiqa", "test"), ("SQuAD", "squad", "dev")]


for dataset_name, dataset, split in datasets:

    print("\n" + "=" * 70)
    print(f"TESTING BM25 ON {dataset_name}")
    print("=" * 70)

    # Load a small subset of the real dataset
    corpus, queries, qrels = load(dataset, split, subset=5, n_extra_docs=50)

    # Convert to the format expected by DocumentDataset
    documents = [{"text": list(corpus.values()), "document_id": list(corpus.keys())}]

    # Build BM25 index
    retriever = BM25(documents, k1=1.2, b=0.75)

    print("Documents indexed:", len(retriever.index.GetDocuments()))
    print("Queries loaded:", len(queries))
    print("Vocabulary size:", len(retriever.corpus_vocabulary))
    print("Average document length:", retriever.average_document_length)

    # Select one real dataset query
    query_id, query_text = next(iter(queries.items()))

    # Calculate BM25 scores
    scores = retriever.CalculateScores([query_text])[0]

    all_documents = retriever.index.GetDocuments()

    # Rank documents by score
    ranked_results = sorted(
        zip(scores, all_documents), key=lambda item: float(item[0]), reverse=True
    )[:5]

    relevant_ids = set(qrels[query_id])

    print("\nSEARCH QUERY:")
    print(query_text)

    print("\nTOP 5 RESULTS:")

    for rank, (score, document) in enumerate(ranked_results, 1):

        doc_id = document.GetId()

        print(f"\nRank: {rank}")
        print(f"Score: {float(score):.4f}")
        print(f"Document ID: {doc_id}")
        print("Relevant:", "YES" if doc_id in relevant_ids else "NO")
        print("Passage:", document.GetText()[:400])

    found = any(document.GetId() in relevant_ids for _, document in ranked_results)

    print("\nRelevant passage found in top 5:", found)


print("\nBOTH BM25 DATASET TESTS FINISHED.")
