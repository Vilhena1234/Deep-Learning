
from src.evaluation.metrics import (
    calculatePrecision,
    calculateRecall,
    calculateReciprocalRank
)
from src.data_utils.query import Query


def test_precision():
    relevancies = [True, False, True, False]
    assert calculatePrecision(relevancies) == 0.5


def test_recall():
    query = Query(
        text="test query",
        id="1",
        relevant_document_ids=["1", "2", "3", "4"]
    )
    relevancies = [True, False, True]
    assert calculateRecall(relevancies, query) == 0.5


def test_reciprocal_rank():
    relevancies = [False, False, True, True]
    assert calculateReciprocalRank(relevancies) == 1 / 3
