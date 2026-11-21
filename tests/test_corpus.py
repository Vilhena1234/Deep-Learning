"""
Tests for the fixes in src/data_utils/corpus.py, using small fake datasets with the
same shape as the real ones (no download needed).
Run from the repo root:  python -m pytest tests/test_corpus.py -v
"""

import os
import sys

import pytest
from datasets import Dataset, DatasetDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import src.data_utils.corpus as corpus_module  # noqa: E402
from src.data_utils.dataset import DocumentDataset  # noqa: E402
from src.data_utils.query import Query  # noqa: E402

# rajpurkar/squad: 'context' is ONE string; several questions share the same paragraph
FAKE_SQUAD = DatasetDict({
    'train': Dataset.from_dict({'question': ['Where is Paris?', 'Capital of France?'],
                                'context': ['Paris is in France.', 'Paris is in France.']}),
    'validation': Dataset.from_dict({'question': ['Where is Berlin?'], 'context': ['Berlin is in Germany.']}),
})
# vibrantlabsai/fiqa "main": 'ground_truths' is a LIST of passages; one passage answers two questions
FAKE_FIQA = DatasetDict({
    'train': Dataset.from_dict({'question': ['q a', 'q b'],
                                'ground_truths': [['answer one', 'shared answer'], ['shared answer']]}),
    'validation': Dataset.from_dict({'question': ['q c'], 'ground_truths': [['answer three']]}),
    'test': Dataset.from_dict({'question': ['q d'], 'ground_truths': [['answer four']]}),
})


@pytest.fixture
def make_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_module, 'get_squad_dataset', lambda base_dir: FAKE_SQUAD)
    monkeypatch.setattr(corpus_module, 'get_fiqa_dataset', lambda base_dir: FAKE_FIQA)
    return lambda name: corpus_module.Corpus(name, str(tmp_path))


def documents_of(corpus):
    return {d.GetId(): d.GetText() for d in DocumentDataset(corpus.all_documents).GetDocuments()}


def test_squad_one_document_per_paragraph(make_corpus):
    corpus = make_corpus('squad')
    docs = documents_of(corpus)
    assert sorted(docs.values()) == ['Berlin is in Germany.', 'Paris is in France.']   # not letters
    # both Paris questions point to the same paragraph
    assert corpus.relevant_ids['q0'] == corpus.relevant_ids['q1'] == [corpus.train['document_id'][0][0]]


def test_fiqa_shared_answer_has_one_id(make_corpus):
    corpus = make_corpus('fiqa')
    docs = documents_of(corpus)
    assert len(docs) == 4                                     # answer one, shared, three, four
    shared_id = next(i for i, t in docs.items() if t == 'shared answer')
    assert shared_id in corpus.relevant_ids['q0'] and corpus.relevant_ids['q1'] == [shared_id]


def test_saved_corpus_loads_the_same(make_corpus):
    first = documents_of(make_corpus('fiqa'))
    second = documents_of(make_corpus('fiqa'))                # second time: read from disk
    assert first == second


def test_works_with_main_py_pipeline(make_corpus):
    """Same steps as main.py, with BM25: every query finds a relevant document at rank 1."""
    from src.retrievers.bm25 import BM25
    from src.retrievers.utils import retrieveQueryAndGetRelevancies

    corpus = make_corpus('squad')
    queries = [Query(corpus.train['question'][i], id=corpus.train['query_id'][i],
                     relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][i]])
               for i in range(corpus.train.num_rows)]
    relevancies = retrieveQueryAndGetRelevancies(BM25(documents=corpus.all_documents), queries, k=1)
    assert relevancies == [[True], [True]]


def test_download_is_not_forced_offline():
    import importlib
    os.environ.pop('HF_DATASETS_OFFLINE', None)
    importlib.reload(importlib.import_module('src.data_utils.download_retrieval_datasets'))
    assert os.environ.get('HF_DATASETS_OFFLINE') is None
