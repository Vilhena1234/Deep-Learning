import torch
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path

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
    with torch.no_grad():
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

    #encoding
    question_vectors = encoder(questions,tokenizer,model)
    passage_vectors = encoder(passages,tokenizer,model)
    print("Question vectors:", question_vectors.shape)
    print("Passage vectors:", passage_vectors.shape)

    #compare questions & passages (vectors)
    scores = question_vectors @ passage_vectors.T
    correct = 0

    #evaluate
    for row, question_id in enumerate(question_ids):
    
        best_position = scores[row].argmax().item()
        predicted_id = passage_ids[best_position]

        is_correct = qrels[question_id].get(predicted_id, 0) > 0
        correct += int(is_correct)

        print("Question:", questions[row])
        print("Predicted passage:", predicted_id)
        print("Correct:", is_correct)
        print()

    print(f"Correct at rank 1: {correct}/{len(question_ids)}")


    

if __name__== "__main__":
    main()