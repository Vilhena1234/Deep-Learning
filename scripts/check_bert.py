"""
Check with the REAL pretrained BERT that our encoder gives the same output.
Downloads bert-base-uncased (~440 MB) the first time.

Run from the repo root:  python scripts/check_bert.py
"""

import os
import sys

import torch
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from src.retrievers.models.bert_loading import load_bert  # noqa: E402

MODEL = "bert-base-uncased"

texts = [
    "What is the tax on dividends?",
    "Dividends are taxed as ordinary income or at the qualified dividend rate.",
    "The river bank was flooded.",
]

ours, tokenizer = load_bert(MODEL)
# "eager" = plain attention code; needed for HuggingFace to return attention maps
bert = BertModel.from_pretrained(MODEL, add_pooling_layer=False, attn_implementation="eager").eval()

batch = tokenizer(texts, padding=True, return_tensors="pt")
ids, mask = batch["input_ids"], batch["attention_mask"]

with torch.no_grad():
    ref = bert(input_ids=ids, attention_mask=mask, output_attentions=True)
    out = ours(ids, mask)

real = mask.bool()                                   # compare real tokens only
diff = (out - ref.last_hidden_state)[real].abs().max().item()
assert ref.attentions is not None and len(ref.attentions) == len(ours.attention_maps())
map_diff = max((o - r).abs().max().item() for o, r in zip(ours.attention_maps(), ref.attentions))

print(f"Texts: {len(texts)}, tokens per text (with padding): {ids.shape[1]}")
print(f"Largest difference in token vectors:   {diff:.2e}")
print(f"Largest difference in attention maps:  {map_diff:.2e}")
print("SAME AS HUGGINGFACE BERT" if diff < 1e-4 and map_diff < 1e-4 else "DIFFERENT - something is wrong")
