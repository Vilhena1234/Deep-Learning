import numpy as np
from src.retrievers.base_retriever import Retriever


class RandomRetriever(Retriever):

    def __init__(self, documents=None, index_path=None, seed=42):
        super().__init__(documents, index_path)
        self.rng = np.random.default_rng(seed)

    def CalculateScores(self, queries: list[str]):
        n_documents = len(self.index.GetDocuments())

        return self.rng.random((len(queries), n_documents))

    def Lookup(self, queries: list[str], k: int):
        if k <= 0:
            raise ValueError("k must be positive")

        documents = list(self.index.GetDocuments())
        k = min(k, len(documents))

        results = []

        for query in queries:
            indices = self.rng.choice(
                len(documents), size=k, replace=False
            )

            results.append([documents[i] for i in indices])

        return results