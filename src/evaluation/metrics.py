from src.data_utils.query import Query




def calculateRecall(relevancies: list[bool], query: Query):
    # return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())
    if query.getNumberOfRelevantDocuments() == 0:
        return 0.0
    return sum(relevancies) / query.getNumberOfRelevantDocuments()

def Evaluate(relevancies: list[list[bool]],
             queries: list[Query], k_values=()):

    assert len(relevancies) == len(queries)

    n_queries = len(queries)

    if n_queries == 0:
        raise ValueError("No queries provided")

    scores = {}
    total_recall = 0
    total_precision = 0
    total_mrr = 0

    for i in range(n_queries):
        total_recall += calculateRecall(
            relevancies[i], queries[i]
        )
        total_precision += calculatePrecision(
            relevancies[i]
        )
        total_mrr += calculateReciprocalRank(
            relevancies[i]
        )

    scores['recall'] = total_recall / n_queries
    scores['precision'] = total_precision / n_queries
    scores['mrr'] = total_mrr / n_queries

    for k in k_values:
        total_precision_at_k = 0
        total_recall_at_k = 0

        for i in range(n_queries):
            total_precision_at_k += calculatePrecisionAtK(
                relevancies[i], k
            )
            total_recall_at_k += calculateRecallAtK(
                relevancies[i], queries[i], k
            )

        scores[f'precision@{k}'] = total_precision_at_k / n_queries
        scores[f'recall@{k}'] = total_recall_at_k / n_queries

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

def calculatePrecisionAtK(relevancies: list[bool], k: int):
    if k <= 0:
        raise ValueError("k must be positive")

    return sum(relevancies[:k]) / k


def calculateRecallAtK(relevancies: list[bool], query: Query, k: int):
    if k <= 0:
        raise ValueError("k must be positive")

    total_relevant = query.getNumberOfRelevantDocuments()

    if total_relevant == 0:
        return 0.0

    return sum(relevancies[:k]) / total_relevant

