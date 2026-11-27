from src.retrievers.bm25 import BM25

# Three small documents for testing
documents = [
    {
        "text": [
            "machine learning machine neural",
            "neural network learning",
            "machine search engine",
        ],
        "document_id": ["D1", "D2", "D3"],
    }
]


# Build the BM25 retriever
retriever = BM25(documents, k1=1.2, b=0.75)


print("\nBM25 PARAMETERS")
print("k1:", retriever.k1)
print("b:", retriever.b)


print("\nVOCABULARY")
print(retriever.sorted_vocabulary)


print("\nDOCUMENT LENGTHS")
print(retriever.document_lengths)

print("Average document length:", retriever.average_document_length)


print("\nBM25 IDF VALUES")

for term, value in sorted(retriever.idf.items()):
    print(f"{term}: {value:.4f}")


print("\nBM25 MATRIX")
print(retriever.bm25_matrix.toarray())

print("\nMATRIX SHAPE")
print(retriever.bm25_matrix.shape)


# Test a search query
query = "neural search"

scores = retriever.CalculateScores([query])[0]


print("\nSEARCH QUERY")
print(query)


print("\nRETRIEVAL SCORES")
print(scores)


# Rank the documents
ranked_results = sorted(
    zip(scores, retriever.index.GetDocuments()),
    key=lambda item: float(item[0]),
    reverse=True,
)


print("\nRANKED DOCUMENTS")

for rank, (score, document) in enumerate(ranked_results, 1):

    print(
        f"Rank {rank}: "
        f"{document.GetId()} - "
        f"{document.GetText()} "
        f"(Score: {float(score):.4f})"
    )


# Basic validation
assert retriever.bm25_matrix.shape == (3, 6)
assert scores.shape == (3,)
assert all(float(score) > 0 for score in scores)

print("\nBM25 BASIC TEST PASSED!")
