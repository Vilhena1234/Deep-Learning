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
def load_ds():
    project_folder = Path(__file__).resolve().parent.parent
    dataset_path = project_folder/"data"/"processed"/"squad_train.json"
    
    with dataset_path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    #question ids
    
    return dataset,question_ids



def create_batch(dataset,question_ids):
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
    
    return questions, passages, passage_ids
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
#tell loss f which pass col is correct for each question
def build_targets(question_ids, passage_ids, qrels, device):
    p_to_col = {
            passage_id: column
            for column, passage_id in enumerate(passage_ids)
        }
    #connecting passage ID to the score columns p_to_col["train_d2"] ->2
        
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
            device=device,
    
        )
    return targets
#how well the model at the time retrives the correct passage
def eval(questions, passages, targets, tokenizer, model):
    was_training = model.training
    model.eval()
    #TRAINING 1 STEP
    #compare questions & passages (vectors)
    #questions as rows and scores of each passsage to the question as columns
    with torch.no_grad():
        question_vecs = encoder(questions, tokenizer,model)
        pass_vecs = encoder(passages,tokenizer,model)
        scores = question_vecs @ pass_vecs.T
        loss = F.cross_entropy(scores,targets)
        pred_cols = scores.argmax(dim=1)
        correct = (pred_cols == targets).sum().item()

    model.train(was_training)
    return loss.item(), correct
def train_step(questions,passages,targets,tokenizer,model,optimizer):
    model.train()
    optimizer.zero_grad()
    q_vecs = encoder(questions,tokenizer,model)
    pass_vecs = encoder(passages,tokenizer,model)
    scores = q_vecs @ pass_vecs.T
    loss = F.cross_entropy(scores,targets)

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


    optimizer.step()
    #evaluate
def main():
    dataset =load_ds()
    question_ids = list(dataset["qrels"])[:20]
    questions, passages, passage_ids = create_batch(dataset,question_ids)

    """
    print("questions:", len(questions))
    print("passages:",len(passages))
    """
    save_dir = (
            Path(__file__).resolve().parent.parent
            /"DPR_training"
            /"checkpoints"
            /"modernbert_squad_5step"
        )
    model_id = "answerdotai/ModernBERT-large"
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModel.from_pretrained(save_dir)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    targets = build_targets(
        question_ids,
        passage_ids,
        dataset["qrels"],
        device=next(model.parameters()).device,
    )
    b4_loss, b4_correct = eval(
        questions, passages, targets, tokenizer, model
    )
    
    step_num = 5
    total = len(questions)
    for step in range(1, step_num+1):
        train_step(
            questions,passages,targets,tokenizer,model,optimizer
        )
        after_loss, after_correct = eval(
            questions, passages, targets, tokenizer, model
        )
        print(
            f"Step {step}: loss={after_loss:.4f}, "
            f"correct={after_correct}/{total}"
        )
        
    print("Total loss decrease:", b4_loss - after_loss)
    print("Additional correct:", after_correct - b4_correct)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Model saved:", save_dir)
    """
    print("Question vectors:", question_vectors.shape)
    print("Passage vectors:", passage_vectors.shape)"""
"""
    print("params inspected:", name)
    print("par changed:",not torch.equal(before,after))
    print("largest change:", max_change )"""
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
"""
print(f"Correct at rank 1: {correct}/{len(question_ids)}")"""
if __name__== "__main__":
    main()