import torch
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import torch.nn.functional as F 

"""
question: question
passage: the text containing the answer

corpus: stores passage IDs and their paragraph texts
queries: stores questions
qrels: identifies which passage is relevant to each question

queries[question ID]  question text
corpus[passage ID]   passage text
qrels[question ID]    relevant passage IDs and scores
"""
#data gathering
def data_loader():
    project_folder = Path(__file__).resolve().parent.parent
    dataset_path = project_folder/"data"/"processed"/"squad_train.json"

    with dataset_path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    #question ids
    question_ids = list(dataset["qrels"])[:20]
    #question text gathering
    questions =[
        dataset["queries"][question_id]
        for question_id in question_ids
    ]
    """
    same as:
    questions = []
    for question_id in question_ids
        question_text = dataset["queries"][question_id]
        questions.append(question_text)
    """
    #passage id gathering
    passage_ids = []
    for question_id in question_ids:
        for passage_id, relevance in dataset["qrels"][question_id].items():
            if relevance>0 and passage_id not in passage_ids:
                passage_ids.append(passage_id)
    #passage text gathering
    passages = [
        dataset["corpus"][passage_id]
        for passage_id in passage_ids
    ]
    
    return questions, passages, passage_ids, question_ids, dataset["qrels"]
    #data gathering end
#encoder
def encoder(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length= 384,
        return_tensors="pt"
        )
    
    outputs = model(**inputs)
    token_vectors=outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    mask = mask.to(token_vectors.dtype)
    vector_sum = (token_vectors*mask).sum(dim=1)
    token_count = mask.sum(dim=1).clamp_min(1)
    text_vectors = vector_sum / token_count


    return text_vectors
def main():
    questions, passages, passage_ids, question_ids, qrels = data_loader()
    print("questions:", len(questions))
    print("passages:",len(passages))


    model_id = "answerdotai/ModernBERT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    #encoding
    question_vectors = encoder(questions,tokenizer,model)
    passage_vectors = encoder(passages,tokenizer,model)
    print("Question vectors:", question_vectors.shape)
    print("Passage vectors:", passage_vectors.shape)

    #compare questions & passages (vectors)
    #questions as rows and scores of each passsage to the question as columns
    scores = question_vectors @ passage_vectors.T
    correct = 0
    #connecting passage ID to the score columns p_to_col["train_d2"] ->2
    p_to_col = {
        passage_id: column
        for column, passage_id in enumerate(passage_ids)
    }
    target_cols =[]
    for question_id in question_ids:
        pos_p_ids = [
            passage_id
            for passage_id,relevance in qrels[question_id].items()
            if relevance > 0 
        ]
        if len(pos_p_ids) != 1:
            raise ValueError("question does not have positive passage")
        #turning pos p ID into target
        pos_p_id = pos_p_ids[0]
        correct_col = p_to_col[pos_p_id]
        target_cols.append(correct_col)

    targets = torch.tensor(
        target_cols,
        dtype=torch.long,
        device=scores.device,

    )
    loss = F.cross_entropy(scores,targets)
    
    #check needed for infinite and NaN
    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is nan or infinite")
    loss.backward()
    #Gradient check b4 update weights

    checked_params = 0
    for name, par in model.named_parameters():
        if par.grad is None:
            continue
        checked_params+=1
        #checking every grad tensor
        if not torch.isfinite(par.grad).all().item():
            raise RuntimeError(f"invalid gradient:{name}")
    if checked_params == 0:
        raise RuntimeError("no param grad calculated")

    print("all grad are good. params checked:", checked_params)


    before = par.detach().clone()
    optimizer.step()
    after = par.detach()
    max_change = (after-before).abs().max().item()
    #evaluate

    print("params inspected:", name)
    print("par changed:",not torch.equal(before,after))
    print("largest change:", max_change )
"""
    print("Passage columns:", p_to_col)
    print("Targets:", targets.tolist())
    print("Score shape:", scores.shape)
    print("Target shape:", targets.shape)
    print("Loss:", loss.item())
    print("Loss tracks gradients:", loss.requires_grad)


   for row, question_id in enumerate(question_ids):
    
        best_position = scores[row].argmax().item()
        predicted_id = passage_ids[best_position]

        is_correct = qrels[question_id].get(predicted_id, 0) > 0
        correct += int(is_correct)

        print("Question:", questions[row])
        print("Predicted passage:", predicted_id)
        print("Correct:", is_correct)
        print()
"""

"""print(f"Correct at rank 1: {correct}/{len(question_ids)}")"""
if __name__== "__main__":
    main()