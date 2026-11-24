import sys
import random
import torch
from transformers import AutoTokenizer, AutoModel
from bert import load_ds, splitter, create_batch, build_targets, encoder


def encode_all(texts, tokenizer, model, batch_size=32):
    
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            chunks.append(encoder([texts[i] for i in idx], tokenizer, model))
    sorted_vecs = torch.cat(chunks)
    vecs = torch.empty_like(sorted_vecs)
    vecs[order] = sorted_vecs         
    return vecs


def evaluate_retrieval(dataset, question_ids, tokenizer, model, batch_size=32, ks=(1, 5, 20)):
    was_training = model.training
    model.eval()
    questions, passages, passage_ids = create_batch(dataset, question_ids)
    device = next(model.parameters()).device
    targets = build_targets(question_ids, passage_ids, dataset["qrels"], device)

    q_vecs = encode_all(questions, tokenizer, model, batch_size)
    p_vecs = encode_all(passages, tokenizer, model, batch_size)
    scores = q_vecs @ p_vecs.T                             
    gold = scores.gather(1, targets.unsqueeze(1))          
    ranks = (scores > gold).sum(dim=1) + 1                 

    metrics = {f"recall@{k}": (ranks <= k).float().mean().item() for k in ks}
    metrics["mrr"] = (1.0 / ranks.float()).mean().item()
    metrics["questions"] = len(question_ids)
    metrics["candidates"] = len(passage_ids)
    model.train(was_training)
    return metrics


def main():
    source = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    dataset = load_ds()
    
    valid_ids, _ = splitter(dataset)
    valid_ids = valid_ids.copy()
    random.Random(37).shuffle(valid_ids)

    if limit is not None:
        valid_ids = valid_ids[:limit]

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(source, evaluate_retrieval(dataset, valid_ids, tokenizer, model))


if __name__ == "__main__":
    main()
