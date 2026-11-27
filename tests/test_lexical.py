"""
Tests for TF-IDF and BM25 on a tiny corpus where every score is worked out by hand.
Run from the repo root:  python -m pytest tests/test_lexical.py -v

Toy corpus (N = 3):
    d1 = "apple apple banana"           counts: apple 2, banana 1     length 3, max count 2
    d2 = "banana cherry"                counts: banana 1, cherry 1    length 2, max count 1
    d3 = "cherry cherry cherry date"    counts: cherry 3, date 1      length 4, max count 3
Document frequencies: apple 1, banana 2, cherry 2, date 1.   Average length = (3+2+4)/3 = 3.
"""

import os
import sys
from math import log

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from src.retrievers.base_retriever import Retriever  # noqa: E402
from src.retrievers.bm25 import BM25  # noqa: E402
from src.retrievers.preprocess import tokenize  # noqa: E402
from src.retrievers.tf_idf import TFIDF  # noqa: E402

CORPUS = {
    "d1": "apple apple banana",
    "d2": "banana cherry",
    "d3": "cherry cherry cherry date",
}


def scores_by_id(retriever, query):
    return dict(retriever.retrieve(query, k=len(CORPUS)))


# ---------------------------------------------------------------- preprocessing
def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Apple, BANANA! cherry?") == ["apple", "banana", "cherry"]


def test_tokenize_options():
    assert tokenize("the dividends of a company", remove_stopwords=True) == ["dividends", "company"]
    assert tokenize("dividends running", stem=True) == ["dividend", "run"]


# ---------------------------------------------------------------- TF-IDF
def test_tfidf_idf_values():
    r = TFIDF.from_corpus_dict(CORPUS)
    assert r.idf["apple"] == pytest.approx(log(3 / 1))
    assert r.idf["banana"] == pytest.approx(log(3 / 2))


def test_tfidf_hand_computed_scores():
    r = TFIDF.from_corpus_dict(CORPUS)
    s = scores_by_id(r, "apple banana")
    # d1: TF(apple) = 2/2, TF(banana) = 1/2
    assert s["d1"] == pytest.approx(1.0 * log(3) + 0.5 * log(3 / 2), rel=1e-5)
    # d2: TF(banana) = 1/1
    assert s["d2"] == pytest.approx(1.0 * log(3 / 2), rel=1e-5)
    # d3 shares no word with the query
    assert s["d3"] == pytest.approx(0.0)


def test_tfidf_ranking_and_output_format():
    r = TFIDF.from_corpus_dict(CORPUS)
    hits = r.retrieve("apple banana", k=2)
    assert [doc_id for doc_id, _ in hits] == ["d1", "d2"]
    assert hits[0][1] >= hits[1][1]  # sorted, best first


def test_unknown_query_word_is_ignored():
    r = TFIDF.from_corpus_dict(CORPUS)
    assert scores_by_id(r, "apple zebra") == pytest.approx(scores_by_id(r, "apple"))


# ---------------------------------------------------------------- BM25
def test_bm25_idf_values():
    r = BM25.from_corpus_dict(CORPUS)
    assert r.idf["apple"] == pytest.approx(log((3 + 1) / (1 + 0.5)))
    assert r.idf["banana"] == pytest.approx(log((3 + 1) / (2 + 0.5)))
    assert r.average_document_length == pytest.approx(3.0)


def test_bm25_hand_computed_scores():
    k1, b = 1.5, 0.75
    r = BM25.from_corpus_dict(CORPUS, k1=k1, b=b)
    s = scores_by_id(r, "apple banana")
    idf_apple, idf_banana = log(4 / 1.5), log(4 / 2.5)

    # d1 has average length (3), so the length factor is 1 - b + b * 3/3 = 1
    tf_apple_d1 = 2 * (k1 + 1) / (2 + k1 * 1)
    tf_banana_d1 = 1 * (k1 + 1) / (1 + k1 * 1)
    assert s["d1"] == pytest.approx(tf_apple_d1 * idf_apple + tf_banana_d1 * idf_banana, rel=1e-5)

    # d2 is shorter (2): length factor = 1 - b + b * 2/3
    tf_banana_d2 = 1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / 3))
    assert s["d2"] == pytest.approx(tf_banana_d2 * idf_banana, rel=1e-5)


def test_bm25_saturation():
    # Each extra repetition adds less, and TF never exceeds k1 + 1.
    r = BM25.from_corpus_dict(CORPUS, k1=1.2, b=0.0)
    tfs = [r.BM25TermFrequency(f, document_length=3) for f in range(1, 50)]
    gains = [b - a for a, b in zip(tfs, tfs[1:])]
    assert all(g2 < g1 for g1, g2 in zip(gains, gains[1:]))
    assert max(tfs) < 1.2 + 1


def test_bm25_b_zero_ignores_length():
    r = BM25.from_corpus_dict(CORPUS, k1=1.2, b=0.0)
    assert r.BM25TermFrequency(2, document_length=1) == pytest.approx(r.BM25TermFrequency(2, document_length=500))


def test_bm25_set_parameters_rebuilds_matrix():
    r = BM25.from_corpus_dict(CORPUS, k1=1.5, b=0.75)
    before = scores_by_id(r, "cherry")
    r.SetParameters(k1=1.5, b=0.0)
    after = scores_by_id(r, "cherry")
    fresh = scores_by_id(BM25.from_corpus_dict(CORPUS, k1=1.5, b=0.0), "cherry")
    assert after == pytest.approx(fresh)
    assert after != pytest.approx(before)


# ---------------------------------------------------------------- works with the repo's pipeline
def test_fast_lookup_matches_base_class_lookup():
    # Our fast Lookup must return the same documents as the starter code's slow version.
    r = BM25.from_corpus_dict(CORPUS)
    queries = ["apple banana", "cherry", "date banana"]
    fast = [[d.GetId() for d in docs] for docs in r.Lookup(queries, k=2)]
    slow = [[d.GetId() for d in docs] for docs in Retriever.Lookup(r, queries, k=2)]
    assert fast == slow
