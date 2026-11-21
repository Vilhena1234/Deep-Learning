from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    # return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())
    if query.getNumberOfRelevantDocuments() == 0:
        return 0.0
    return sum(relevancies) / query.getNumberOfRelevantDocuments()

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {
    'recall': None,
    'precision': None,
    'mrr': None
}
    n_queries = len(queries)
    total_recall = 0
    total_precision = 0
    total_mrr = 0  

    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i])
        total_mrr += calculateReciprocalRank(relevancies[i])
    scores['recall'] = total_recall / n_queries
    scores['precision'] = total_precision / n_queries
    scores['mrr'] = total_mrr / n_queries   
    return scores
def calculatePrecision(relevancies: list[bool]):
    if len(relevancies) == 0:
        return 0.0

    return sum(relevancies) / len(relevancies)


def calculateReciprocalRank(relevancies: list[bool]):
    for rank, relevant in enumerate(relevancies, start=1):
        if relevant:
            return 1 / rank

    return 0.0