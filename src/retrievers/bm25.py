from collections import Counter
from math import log

from scipy.sparse import lil_matrix

from src.retrievers.tf_idf import TFIDF


class BM25(TFIDF):

    def __init__(
        self,
        documents: list[dict] = None,
        index_path: str = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:

        super(BM25, self).__init__(documents, index_path)

        # BM25 parameters
        self.k1 = k1
        self.b = b

        if self.k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")

        # Calculate document lengths and average document length
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()

        # Build the BM25 document scoring matrix
        self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Calculate BM25 inverse document frequencies.

        IDF = log((N + 1) / (DF + 0.5))
        """

        document_frequencies = self.GetDocumentFrequencies()

        n_documents = len(self.index.GetDocuments())

        if n_documents == 0:
            return {}

        idf = {}

        for term, frequency in document_frequencies.items():

            idf[term] = log((n_documents + 1) / (frequency + 0.5))

        return idf

    def GetDocumentLengths(self):
        """
        Calculate the length of every document
        and the average document length.

        Returns:
            document_lengths: List of document lengths.
            average_document_length: Average length.
        """

        document_lengths = []

        for document in self.index.GetDocuments():

            tokens = self.PreprocessText(document.GetText())

            document_lengths.append(len(tokens))

        if not document_lengths:
            return [], 0

        average_document_length = sum(document_lengths) / len(document_lengths)

        return document_lengths, average_document_length

    def GetDocumentsBM25Vectors(self):
        """
        Build the BM25 document-term matrix.

        Rows = Documents
        Columns = Vocabulary terms
        Values = BM25 weights
        """

        n_documents = len(self.index.GetDocuments())

        n_terms = len(self.sorted_vocabulary)

        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=float)

        if n_documents == 0 or self.average_document_length == 0:
            return bm25_matrix.tocsr()

        # Process each document
        for doc_idx, document in enumerate(self.index.GetDocuments()):

            # Preprocess document and count term occurrences
            tokens = self.PreprocessText(document.GetText())

            term_counts = Counter(tokens)

            document_length = self.document_lengths[doc_idx]

            # Document length normalisation
            length_normalisation = self.k1 * (
                1 - self.b + self.b * (document_length / self.average_document_length)
            )

            # Calculate BM25 weight for every term
            for term, frequency in term_counts.items():

                term_idx = self.term_to_index[term]

                numerator = frequency * (self.k1 + 1)

                denominator = frequency + length_normalisation

                bm25_score = self.idf[term] * numerator / denominator

                bm25_matrix[doc_idx, term_idx] = bm25_score

        return bm25_matrix.tocsr()

    def CalculateScores(self, queries: list[str]):
        """
        Calculate BM25 scores for a list of queries.

        Returns:
            Array with shape:
            (number of queries, number of documents)
        """

        # Convert queries into numerical vectors
        query_vectors = self.QueryToVector(queries)

        # Calculate query-document BM25 scores
        scores = query_vectors.dot(self.bm25_matrix.T).toarray()

        return scores
