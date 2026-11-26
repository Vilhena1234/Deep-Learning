import torch
from transformers import get_linear_schedule_with_warmup
import json
from pathlib import Path
import torch.nn.functional as F 
import random
import time
import sys
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
from src.retrievers.models.dpr_encoder import load_encoder, save_encoder


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
    
    return dataset

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
#splitting ds to by passage for train and validation
def splitter(dataset, validation_percent=0.1, seed = 37):
    tv_groups = {}
    for question_id, relevance_num in dataset["qrels"].items():
        positive_ids = [
            passage_id
            for passage_id, relevance in relevance_num.items()
            if relevance>0
        ]
        if len(positive_ids) !=1:
            raise ValueError("1 pos passage expected")
        
        passage_id = positive_ids[0]
        if passage_id not in tv_groups:
            tv_groups[passage_id] = []

        tv_groups[passage_id].append(question_id)
        #exclude already trained
    trained_already = {"train_d0", "train_d1", "train_d2", "train_d3"}
    untrained_pass = [
        passage_id
        for passage_id in tv_groups
        if passage_id not in trained_already
    ]

    random.Random(seed).shuffle(untrained_pass)
    validation_10 = round(len(tv_groups)*validation_percent)
    validation_set = set(untrained_pass[:validation_10])

    train_ids = []
    valid_ids = []
    for passage_id, question_id in tv_groups.items():
        if passage_id in validation_set:
            valid_ids.extend(question_id)
        else:
            train_ids.extend(question_id)
    
    return valid_ids, train_ids
#encoder
def encoder(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length= 384,
        return_tensors="pt"
        )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    text_vectors = model(inputs["input_ids"], inputs["attention_mask"])


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

    #print("all grad are good. params checked:", checked_params)


    optimizer.step()
    return loss.item()
    #evaluate
def main():
    torch.manual_seed(37)
    source_model = sys.argv[1] if len(sys.argv) > 1 else "own:bert-base-uncased"
    pooling_type = sys.argv[2] if len(sys.argv) > 2 else "mean"
    run_name = sys.argv[3] if len(sys.argv) > 3 else "own_bert_mean_bs16"
    print("model:", source_model, "pooling:", pooling_type, "run:", run_name)


    output_dir = (
            Path(__file__).resolve().parent
            /"checkpoints"
            /"own_bert_mean_bs16"
            )
       
    model, tokenizer = load_encoder(source_model, pooling_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Device used:",device)
    
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    dataset =load_ds()
    valid_ids, train_ids = splitter(dataset)
    batch_size = 16
    
    num_epoch = 1
    train_limit = None
    train_copy = train_ids.copy()
    random.Random(37).shuffle(train_copy)
    if train_limit is not None:
        train_copy = train_copy[:train_limit]

    steps_in_epoch = len(train_copy) // batch_size
    total_steps = steps_in_epoch * num_epoch

    #LR scheduler

    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps= int(0.1 * total_steps),
        num_training_steps=total_steps,
    ) 

    print("questions:", len(train_copy), "steps/epoch:", steps_in_epoch, "total:", total_steps)
    skipped = 0
    step = 0
    start_time = time.time()
    for epoch in range(1, num_epoch + 1):
        random.Random(37).shuffle(train_copy)
        for start in range(0, steps_in_epoch * batch_size, batch_size):
            batch_ids = train_copy[start:start + batch_size]

            questions, passages, passage_ids = create_batch(
                dataset, batch_ids
            )
            #when noting to compare to
            if len(passages) < 2:
                skipped+=1
                continue

            targets = build_targets(
                    batch_ids,
                    passage_ids,
                    dataset["qrels"],
                    device=next(model.parameters()).device,
                )
            
            lr_now = scheduler.get_last_lr()[0]

            train_loss = train_step(
                questions,passages,targets,tokenizer,model,optimizer
            )
            scheduler.step()
            step +=1
            print(f"epoch {epoch} Step {step}: training loss={train_loss:.6f}, lr={lr_now:.2e}")
            
        """
        print("questions:", len(questions))
        print("passages:",len(passages))
        """
        elapsed = time.time() - start_time
        print("total training time:", elapsed)

        if step > 0:
            print("time/step:", elapsed / step)
        if device == "cuda":
            print("peak GPU memory:", torch.cuda.max_memory_allocated() / 1e9, "GB")

        
        epoch_save = output_dir / f"epoch_{epoch}"
        save_encoder(model, tokenizer, epoch_save)

        print(f"Saved epoch {epoch} to {epoch_save}   (skipped batches so far: {skipped})")
    
    
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