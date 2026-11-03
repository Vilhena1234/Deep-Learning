import torch
from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

text = [
    "What is the capital of France?",      
    "Paris is the capital of France.",     
    "Berlin is the capital of Germany.",  
    "Apples grow on trees.",              
]
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_lenght=128,
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



# Predicted token:  Paris
print("Token vectors:", token_vectors.shape)
print("Whole-text vectors:", text_vectors.shape)

query_vector = text_vectors[0]  
passage_vectors = text_vectors[1:]

scores = passage_vectors @ query_vector

ranking = torch.argsort(scores, descending=True)

for position in ranking.tolist():
    print(f"Score: {scores[position].item():.3f}")
    print(f"Passage: {text[position + 1]}")
    print()