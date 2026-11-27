from collections import Counter
from math import log

import numpy as np
from scipy.sparse import lil_matrix

from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
from src.retrievers.preprocessing import preprocess_text


class TFIDF(Retriever):

    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:

        super(TFIDF, self).__init__(documents, index_path)

        # Build the vocabulary from the complete corpus
        self.corpus_vocabulary = self.GetCorpusVocabulary()

        # Create a consistent vocabulary ordering
        self.sorted_vocabulary = sorted(self.corpus_vocabulary)

        # Create a mapping between terms and matrix column positions
        # This mapping is shared by TF-IDF and BM25
        self.term_to_index = {
            term: idx for idx, term in enumerate(self.sorted_vocabulary)
        }

        # Calculate inverse document frequencies
        self.idf = self.GetInverseDocumentFrequencies()

        # Build the TF-IDF matrix only for the TFIDF class itself.
        # BM25 inherits this class but builds its own scoring matrix.
        if type(self) is TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Preprocess text using the shared preprocessing module.
        """
        return preprocess_text(text)

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Return all unique words in a query.
        """
        tokens = self.PreprocessText(query)
        return sorted(set(tokens))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Return all unique words in a document.
        """
        tokens = self.PreprocessText(document.GetText())
        return sorted(set(tokens))

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Return all unique words across the entire corpus.
        """

        corpus_vocabulary = set()

        for document in self.index.GetDocuments():

            tokens = self.PreprocessText(document.GetText())

            corpus_vocabulary.update(tokens)

        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Calculate document frequency (DF).

        DF represents the number of documents
        containing each term.
        """

        document_frequencies = {term: 0 for term in self.corpus_vocabulary}

        for document in self.index.GetDocuments():

            document_terms = self.GetDocumentVocabulary(document)

            for term in document_terms:

                document_frequencies[term] += 1

        return document_frequencies

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Calculate IDF for every term.

        IDF = log(N / DF)

        N  = Total number of documents
        DF = Number of documents containing the term
        """

        document_frequencies = self.GetDocumentFrequencies()

        n_documents = len(self.index.GetDocuments())

        if n_documents == 0:
            return {}

        idf = {}

        for term, frequency in document_frequencies.items():

            if frequency <= 0:
                raise ValueError(f"Invalid document frequency for term: {term}")

            idf[term] = log(n_documents / frequency)

        return idf

    def GetDocumentTermCounts(self, document: Document) -> dict[str, int]:
        """
        Count the occurrences of every term in a document.
        """

        tokens = self.PreprocessText(document.GetText())

        term_counts = Counter(tokens)

        return dict(term_counts)

    def GetDocumentsTFIDFVectors(self):
        """
        Build the document-term TF-IDF matrix.

        Rows    = Documents
        Columns = Vocabulary terms
        Values  = TF-IDF weights

        TF = Term frequency / Maximum term frequency
        IDF = log(N / DF)
        """

        n_documents = len(self.index.GetDocuments())

        n_terms = len(self.sorted_vocabulary)

        # Initialise sparse document-term matrix
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # Process every document
        for doc_idx, document in enumerate(self.index.GetDocuments()):

            # Count term occurrences
            term_counts = self.GetDocumentTermCounts(document)

            # Find maximum term frequency in this document
            max_tf = max(term_counts.values(), default=0)

            if max_tf == 0:
                continue

            # Calculate TF-IDF weights
            for term, count in term_counts.items():

                tf = count / max_tf

                idf = self.idf[term]

                tfidf_score = tf * idf

                # Find the correct vocabulary column
                term_idx = self.term_to_index[term]

                # Store the TF-IDF weight
                tfidf_matrix[doc_idx, term_idx] = tfidf_score

        # Convert to CSR for efficient matrix operations
        return tfidf_matrix.tocsr()

    def QueryToVector(self, queries: list[str]):
        """
        Convert queries into sparse numerical vectors
        using the corpus vocabulary.

        Each entry represents the frequency of a query term.
        """

        query_vector = lil_matrix(
            (len(queries), len(self.sorted_vocabulary)), dtype=np.float32
        )

        for query_idx, query in enumerate(queries):

            # Preprocess and count query terms
            query_counts = Counter(self.PreprocessText(query))

            # Add terms that exist in the corpus vocabulary
            for term, count in query_counts.items():

                if term in self.term_to_index:

                    term_idx = self.term_to_index[term]

                    query_vector[query_idx, term_idx] = count

        return query_vector.tocsr()

    def CalculateScores(self, queries: list[str]):
        """
        Calculate TF-IDF retrieval scores.

        Returns:
            NumPy array with shape:
            (number of queries, number of documents)
        """

        # Convert queries into numerical vectors
        query_vectors = self.QueryToVector(queries)

        # Calculate query-document similarity scores
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray()

        return scores
