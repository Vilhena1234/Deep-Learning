import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.random_retriever import RandomRetriever
from src.data_utils.query import Query


from src.evaluation.evaluate import evaluate_retriever, save_results
from pathlib import Path

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    print("Total corpus documents:", len(corpus.all_documents))
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
   
    ## Initialize RandomRetriever
    random_model = RandomRetriever(documents=corpus.all_documents, seed=42)
    print("Total indexed documents:", len(random_model.index.GetDocuments()))
      ### Evaluate
    k_values = [1, 5, 10, 100]

    train_scores = evaluate_retriever(
        random_model, train_queries, k_values
    )

    val_scores = evaluate_retriever(
        random_model, val_queries, k_values
    )

    test_scores = evaluate_retriever(
        random_model, test_queries, k_values
    )

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')
    results = [
        {
            "dataset": cfg.dataset.name,
            "model": "RandomRetriever",
            "split": "train",
            "n_queries": len(train_queries),
            **train_scores
        },
        {
            "dataset": cfg.dataset.name,
            "model": "RandomRetriever",
            "split": "val",
            "n_queries": len(val_queries),
            **val_scores
        },
        {
            "dataset": cfg.dataset.name,
            "model": "RandomRetriever",
            "split": "test",
            "n_queries": len(test_queries),
            **test_scores
        }
    ]

    save_results(
        results,
        Path(__file__).resolve().parent / "results/random_fiqa_full.csv"
    )

if __name__ == '__main__':
    main()