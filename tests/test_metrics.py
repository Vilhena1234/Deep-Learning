
from src.evaluation.metrics import (
    calculatePrecision,
    calculateRecall,
    calculateReciprocalRank,
    calculatePrecisionAtK,
    calculateRecallAtK
)

from src.data_utils.query import Query

from src.evaluation.metrics import Evaluate

import pytest

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

def test_precision_at_k():
    relevancies = [True, False, True, False, True]

    assert calculatePrecisionAtK(relevancies, 1) == 1.0
    assert calculatePrecisionAtK(relevancies, 3) == 2 / 3
    assert calculatePrecisionAtK(relevancies, 5) == 3 / 5


def test_recall_at_k():
    query = Query(
        text="test query",
        id="1",
        relevant_document_ids=["1", "2", "3", "4"]
    )

    relevancies = [True, False, True, False, True]

    assert calculateRecallAtK(relevancies, query, 1) == 1 / 4
    assert calculateRecallAtK(relevancies, query, 3) == 2 / 4
    assert calculateRecallAtK(relevancies, query, 5) == 3 / 4
def test_evaluate():
    queries = [
        Query(
            text="first query",
            id="1",
            relevant_document_ids=["1", "2"]
        ),
        Query(
            text="second query",
            id="2",
            relevant_document_ids=["3"]
        )
    ]

    relevancies = [
        [True, False, True] + [False] * 97,
        [False, True, False] + [False] * 97
    ]

    scores = Evaluate(
        relevancies, queries, [1, 5, 10, 100]
    )

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
    assert scores["precision@1"] == pytest.approx(0.5)
    assert scores["precision@5"] == pytest.approx(0.3)
    assert scores["precision@10"] == pytest.approx(0.15)
    assert scores["precision@100"] == pytest.approx(0.015)

    assert scores["recall@1"] == pytest.approx(0.25)
    assert scores["recall@5"] == pytest.approx(1.0)
    assert scores["recall@10"] == pytest.approx(1.0)
    assert scores["recall@100"] == pytest.approx(1.0)

    assert scores["mrr"] == pytest.approx(0.75)