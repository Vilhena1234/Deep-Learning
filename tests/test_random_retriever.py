from types import SimpleNamespace
from src.retrievers.random_retriever import RandomRetriever


def test_random_retriever(monkeypatch):
    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    monkeypatch.setattr(
        "src.retrievers.base_retriever.DocumentDataset",
        lambda docs: SimpleNamespace(
            GetDocuments=lambda: docs
        )
    )

    model = RandomRetriever(documents=documents, seed=42)

    results = model.Lookup(
        queries=["query1", "query2"], k=3
    )

    assert len(results) == 2

    for result in results:
        assert len(result) == 3
        assert len(set(result)) == 3
        assert all(doc in documents for doc in result)