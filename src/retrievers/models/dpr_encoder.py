from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.retrievers.models.bert_loading import load_bert_encoder
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.pooling import pool

WEIGHTS_FILE = "own_encoder.pt"
SETTINGS = ["n_layers", "n_heads", "emb_dim", "hidden_dim", "feed_forward_dim",
            "vocab_size", "pooling_type", "dropout", "max_len", "positions"]


class HFEncoder(nn.Module):
    def __init__(self, hf_model, pooling_type="mean"):
        super().__init__()
        self.hf_model = hf_model
        self.pooling_type = pooling_type

    def forward(self, input_ids, attention_mask):
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        return pool(outputs.last_hidden_state, attention_mask, self.pooling_type)


def load_encoder(source, pooling_type="mean"):
    source = str(source)
    weights_path = Path(source) / WEIGHTS_FILE

    if weights_path.exists():
        saved = torch.load(weights_path, map_location="cpu", weights_only=True)
        model = TransformerEncoder(**saved["settings"])
        model.load_state_dict(saved["weights"])
        tokenizer = AutoTokenizer.from_pretrained(source)
    elif source.startswith("own:"):
        model, tokenizer = load_bert_encoder(source.removeprefix("own:"), pooling_type)
    else:
        model = HFEncoder(AutoModel.from_pretrained(source), pooling_type)
        tokenizer = AutoTokenizer.from_pretrained(source)

    return model.eval(), tokenizer


def save_encoder(model, tokenizer, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(model, HFEncoder):
        model.hf_model.save_pretrained(folder)
    else:
        settings = {name: getattr(model, name) for name in SETTINGS}
        torch.save({"settings": settings, "weights": model.state_dict()}, folder / WEIGHTS_FILE)
    tokenizer.save_pretrained(folder)
