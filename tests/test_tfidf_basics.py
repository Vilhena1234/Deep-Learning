from src.retrievers.tf_idf import TFIDF

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


# Initialise TF-IDF
model = TFIDF(documents=documents)


# 1. Check vocabulary
print("\nVOCABULARY")
print(sorted(model.corpus_vocabulary))


# 2. Check document frequencies
print("\nDOCUMENT FREQUENCIES")
print(model.GetDocumentFrequencies())


# 3. Check IDF
print("\nINVERSE DOCUMENT FREQUENCIES")

for term, value in sorted(model.idf.items()):
    print(f"{term}: {value:.3f}")


# 4. Check term counts for D1
print("\nTERM COUNTS FOR D1")

first_document = model.index.GetDocuments()[0]

print(model.GetDocumentTermCounts(first_document))
print("\nTF-IDF MATRIX")
print(model.tfidf_matrix.toarray())

print("\nMATRIX SHAPE")
print(model.tfidf_matrix.shape)
# 5. Check query vector
print("\nQUERY VECTOR")

queries = [input("\nEnter your search query: ")]

query_vectors = model.QueryToVector(queries)

print(query_vectors.toarray())


# 6. Calculate relevance scores
print("\nRETRIEVAL SCORES")

scores = model.CalculateScores(queries)

print(scores)


# 7. Retrieve the top documents
print("\nRANKED DOCUMENTS")

results = model.Lookup(queries, k=3)

for rank, document in enumerate(results[0], start=1):
    print(f"Rank {rank}: {document.GetId()} - {document.GetText()}")
