"""
TF-IDF retriever, implemented from scratch (Person 2 - Soni).

Formulas (project PDF, section 2.1):
    TF(q, D)  = f(q, D) / max_k f(k, D)      f = how often a word occurs in document D
    IDF(q)    = log(N / n_q)                 N = number of documents, n_q = documents containing q
    Score(Q, D) = sum over query words q of TF(q, D) * IDF(q)

Idea: we precompute ONE big sparse matrix, rows = documents, columns = words,
cell = TF * IDF. A query becomes a row vector of word counts, and the scores
for all documents are a single matrix product:  query_vector @ matrix.T
"""

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
from src.retrievers.preprocess import tokenize


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None,
                 remove_stopwords: bool = False, stem: bool = False) -> None:
        # Preprocessing options must be set before the parent builds anything.
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._doc_tokens = None  # cache: tokenizing 57k passages once is enough

        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        # Fixed word -> column number mapping (sorted, so it is the same on every run).
        self.term_to_index = {term: idx for idx, term in enumerate(sorted(self.corpus_vocabulary))}
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class (BM25 builds its own matrix)
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    @classmethod
    def from_corpus_dict(cls, corpus: dict[str, str], **kwargs):
        """Build from data.py's format: corpus = {doc_id: text}."""
        return cls(documents=[{'text': list(corpus.values()), 'document_id': list(corpus.keys())}], **kwargs)

    # ------------------------------------------------------------------ #
    # Step 1: text -> words
    # ------------------------------------------------------------------ #
    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        return tokenize(text, remove_stopwords=self.remove_stopwords, stem=self.stem)

    def _DocumentTokens(self) -> list[list[str]]:
        """Tokenized text of every document, computed once and reused."""
        if self._doc_tokens is None:
            self._doc_tokens = [self.PreprocessText(doc.GetText()) for doc in self.index.GetDocuments()]
        return self._doc_tokens

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return list(dict.fromkeys(self.PreprocessText(query)))  # unique, original order kept

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return list(dict.fromkeys(self.PreprocessText(document.GetText())))

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        return set().union(*map(set, self._DocumentTokens()))

    # ------------------------------------------------------------------ #
    # Step 2: statistics over the whole collection
    # ------------------------------------------------------------------ #
    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        document_frequencies = Counter()
        for tokens in self._DocumentTokens():
            document_frequencies.update(set(tokens))  # set(): count each word once per document
        return dict(document_frequencies)

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        IDF(q) = log(N / n_q)
        """
        N = len(self.index.GetDocuments())
        return {term: np.log(N / n_q) for term, n_q in self.GetDocumentFrequencies().items()}

    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        return dict(Counter(self.PreprocessText(document.GetText())))

    # ------------------------------------------------------------------ #
    # Step 3: the document x word matrix
    # ------------------------------------------------------------------ #
    def _BuildMatrix(self, cell_value) -> csr_matrix:
        """
        Shared by TF-IDF and BM25. For every (document j, word t, count f) in the corpus,
        cell_value(f, max_f, doc_length, t) gives the number stored at row j, column t.
        Only words that occur in a document are stored (sparse matrix), which is why
        57k documents x 100k words fits in memory.
        """
        rows, cols, values = [], [], []
        for j, tokens in enumerate(self._DocumentTokens()):
            counts = Counter(tokens)
            if not counts:
                continue  # empty document: whole row stays 0
            max_f = max(counts.values())
            for term, f in counts.items():
                rows.append(j)
                cols.append(self.term_to_index[term])
                values.append(cell_value(f, max_f, len(tokens), term))

        shape = (len(self.index.GetDocuments()), len(self.term_to_index))
        return csr_matrix((np.array(values, dtype=np.float32), (rows, cols)), shape=shape)

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        return self._BuildMatrix(lambda f, max_f, length, term: (f / max_f) * self.idf[term])

    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.

        Args:
            query (str): The query to convert

        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        rows, cols, values = [], [], []
        for i, query in enumerate(queries):
            for term, count in Counter(self.PreprocessText(query)).items():
                if term in self.term_to_index:  # words never seen in the corpus cannot match anything
                    rows.append(i)
                    cols.append(self.term_to_index[term])
                    values.append(count)
        shape = (len(queries), len(self.term_to_index))
        return csr_matrix((np.array(values, dtype=np.float32), (rows, cols)), shape=shape)

    # ------------------------------------------------------------------ #
    # Step 4: score and rank
    # ------------------------------------------------------------------ #
    def _ScoreMatrix(self):
        return self.tfidf_matrix

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query

        Args:
            query (str): The query to calculate scores for

        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D)

        """
        query_vectors = self.QueryToVector(queries)  # n_queries x vocab size
        scores = query_vectors.dot(self._ScoreMatrix().T).toarray()  # n_queries x n_documents
        return scores

    def TopK(self, queries: list[str], k: int, batch_size: int = 256) -> list[list[tuple[int, float]]]:
        """
        Fast ranking: for each query, (document position, score) of the k best documents.
        Queries are scored in batches so we never hold a 5,500 x 57,638 score table in memory,
        and np.argpartition finds the top k without sorting all 57k scores.
        """
        results = []
        n_docs = len(self.index.GetDocuments())
        k = min(k, n_docs)
        for start in range(0, len(queries), batch_size):
            scores = self.CalculateScores(queries[start:start + batch_size])
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]  # the k best, unordered
            for row, idx in zip(scores, top):
                idx = idx[np.argsort(-row[idx], kind='stable')]  # order those k by score
                results.append([(int(j), float(row[j])) for j in idx])
        return results

    def Lookup(self, queries: list[str], k: int) -> list:
        """
        Same output as the base class (for each query, a list of the k best Document objects),
        but fast enough for the full FiQA corpus.
        """
        documents = self.index.GetDocuments()
        return [[documents[j] for j, _ in hits] for hits in self.TopK(queries, k)]

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Interface from the work division plan: [(doc_id, score), ...], best first."""
        documents = self.index.GetDocuments()
        return [(documents[j].GetId(), score) for j, score in self.TopK([query], k)[0]]
