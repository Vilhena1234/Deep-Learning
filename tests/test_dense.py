import pytest
import torch

from src.retrievers.base_retriever import Retriever
from src.retrievers.dense import DenseRetriever

TOY = {"q1": [1., 2.], "q2": [3., 0.], "p1": [3., 1.], "p2": [0., 4.], "p3": [2., 0.]}
DOCUMENTS = [{"text": ["p1", "p2", "p3"], "document_id": ["d1", "d2", "d3"]}]


class ToyRetriever(DenseRetriever):
    """uses hand-made vectors instead of a real model."""
    def __init__(self, documents, model_path="toy"):
        Retriever.__init__(self, documents)
        self.model_path = model_path
        self.device = "cpu"
        self.doc_vectors = self.Encode([d.GetText() for d in self.index.GetDocuments()])

    def Encode(self, texts):
        return torch.tensor([TOY[text] for text in texts])


def test_scores_match_hand_calculation():
    scores = ToyRetriever(DOCUMENTS).CalculateScores(["q1", "q2"])
    assert scores.tolist() == [[5, 8, 2], [9, 0, 6]]


def test_topk_and_lookup_order():
    retriever = ToyRetriever(DOCUMENTS)
    assert retriever.TopK(["q1"], 2) == [[(1, 8.0), (0, 5.0)]]
    assert [d.GetId() for d in retriever.Lookup(["q2"], 3)[0]] == ["d1", "d3", "d2"]
    assert len(retriever.TopK(["q1"], 100)[0]) == 3


def test_save_and_load(tmp_path):
    path = tmp_path / "index" / "vectors.pt"
    first = ToyRetriever(DOCUMENTS)
    first.SaveVectors(path)
    second = ToyRetriever(DOCUMENTS)
    second.doc_vectors = None
    second.LoadVectors(path)
    assert torch.equal(first.doc_vectors, second.doc_vectors)


def test_load_rejects_wrong_files(tmp_path):
    path = tmp_path / "vectors.pt"
    ToyRetriever(DOCUMENTS).SaveVectors(path)
    other = [{"text": ["p1"], "document_id": ["d9"]}]
    with pytest.raises(ValueError):
        ToyRetriever(other).LoadVectors(path)
    with pytest.raises(ValueError):
        ToyRetriever(DOCUMENTS, model_path="another").LoadVectors(path)
