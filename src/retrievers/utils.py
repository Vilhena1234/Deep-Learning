from src.retrievers.base_retriever import Retriever
from src.data_utils.query import Query

def retrieveQueryAndGetRelevancies(model: Retriever, queries: list[Query], k: int):
    """
    @param model: Retrieval model
    @param queries: A list of queries.
    @param k: Top-k relevant documents to consider.

    Returns a list of lists of booleans that are True when the retrieved document matches the ground truth of that query.
    """
    N = len(queries)
    retrieved_documents = model.Lookup(queries=[query.getQuery() for query in queries], k=k)
    relevancies = []
    for i in range(N):
        query_relevancies = []
        for document in retrieved_documents[i]:
            if queries[i].isDocumentRelevant(document):
                query_relevancies.append(True)
            else:
                query_relevancies.append(False)
        relevancies.append(query_relevancies)
    return relevancies