"""
BM25 retriever, implemented from scratch (Person 2 - Soni).

BM25 = TF-IDF with two fixes (project PDF, section 2.1):

  1. Saturation: the 10th repetition of a word adds much less than the 1st.
  2. Length normalization: long documents are scaled down, so they do not win
     just because they contain more words.

    TF(q, D)  = f * (k1 + 1) / ( f + k1 * (1 - b + b * |D| / D_avg) )
    IDF(q)    = log( (N + 1) / (n_q + 0.5) )

    k1 (kappa in the PDF): how slowly TF saturates. Larger k1 = closer to raw counts.
    b in [0, 1]: how strongly to normalize length. b = 0 switches it off.

Everything else (tokenizing, vocabulary, sparse matrix, fast ranking) is inherited from TFIDF.
"""

import numpy as np

from src.retrievers.tf_idf import TFIDF


class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75,
                 remove_stopwords: bool = False, stem: bool = False) -> None:
        super(BM25, self).__init__(documents, index_path, remove_stopwords=remove_stopwords, stem=stem)
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()
        self.SetParameters(k1, b)

    def GetDocumentLengths(self):
        """|D| for every document (number of words after preprocessing) and their average D_avg."""
        document_lengths = {}
        average_document_length = 0
        for document, tokens in zip(self.index.GetDocuments(), self._DocumentTokens()):
            document_lengths[document] = len(tokens)
            average_document_length += len(tokens)
        return document_lengths, average_document_length / len(self.index.GetDocuments())

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        BM25 version: IDF(q) = log((N + 1) / (n_q + 0.5)).
        Python calls this instead of the TF-IDF version, because BM25 overrides it.
        """
        N = len(self.index.GetDocuments())
        return {term: np.log((N + 1) / (n_q + 0.5)) for term, n_q in self.GetDocumentFrequencies().items()}

    def BM25TermFrequency(self, f: int, document_length: int) -> float:
        """Saturated, length-normalized term frequency for one word in one document."""
        length_ratio = document_length / self.average_document_length
        return f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length_ratio))

    def GetDocumentsBM25Vectors(self):
        """n_documents x n_terms matrix whose element (i, j) is TF_BM25 * IDF_BM25 of term j in document i."""
        return self._BuildMatrix(lambda f, max_f, length, term: self.BM25TermFrequency(f, length) * self.idf[term])

    def SetParameters(self, k1: float, b: float) -> None:
        """
        Change k1 and b without re-reading the corpus (only the matrix is rebuilt).
        Used for the k1 / b grid search.
        """
        assert k1 >= 0 and 0 <= b <= 1, 'need k1 >= 0 and b in [0, 1]'
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def _ScoreMatrix(self):
        return self.bm25_matrix

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query

        Args:
            query (str): The query to calculate scores for

        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores
