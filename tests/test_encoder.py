"""
Tests for the course's TransformerEncoder (src/retrievers/models/encoder.py)
and max pooling. Run from the repo root:  python -m pytest tests -v
"""

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf
from transformers import BertConfig, BertModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from src.retrievers.models.bert_loading import copy_bert_weights  # noqa: E402
from src.retrievers.models.encoder import TransformerEncoder  # noqa: E402
from src.retrievers.models.pooling import max_pooling  # noqa: E402

torch.manual_seed(0)

# Same sizes as config/model/encoder.yaml
CFG = dict(n_layers=6, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768)


def batch(B=2, T=7, vocab=30522):
    ids = torch.randint(0, vocab, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, -3:] = 0                                    # 3 padding tokens in text 2
    return ids, mask


# --------------------------------------------------------------------------- #
# Building the model
# --------------------------------------------------------------------------- #
def test_builds_with_course_config_and_returns_one_vector_per_text():
    enc = TransformerEncoder(**CFG).eval()
    ids, mask = batch()
    out = enc(ids, mask)
    assert out.shape == (2, 128)                        # batch_size x emb_dim


def test_from_yaml_config():
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "config", "model", "encoder.yaml"))
    enc = TransformerEncoder.from_config(cfg).eval()
    assert enc.pooling_type == cfg.pooling_type and enc.emb_dim == cfg.emb_dim
    assert len(enc.model.layers) == cfg.n_layers


def test_wrong_hidden_dim_gives_clear_error():
    with pytest.raises(AssertionError, match="n_heads \\* hidden_dim"):
        TransformerEncoder(n_layers=2, n_heads=4, emb_dim=128, hidden_dim=16, feed_forward_dim=256)


def test_attention_maps_one_per_layer():
    enc = TransformerEncoder(**CFG).eval()
    ids, mask = batch()
    enc(ids, mask)
    maps = enc.attention_maps()
    assert len(maps) == 6 and maps[0].shape == (2, 4, 7, 7)   # layers x (B, heads, T, T)


def test_trainable_end_to_end():
    """Gradients reach the first layer (embeddings), so DPR training can update everything."""
    enc = TransformerEncoder(**CFG)
    ids, mask = batch()
    enc(ids, mask).sum().backward()
    assert enc.model.embeddings.tok.weight.grad is not None


# --------------------------------------------------------------------------- #
# Pooling types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", ["cls", "mean", "max"])
def test_padding_does_not_change_the_text_vector(pooling):
    """The same text gives the same vector alone or padded inside a batch."""
    enc = TransformerEncoder(**CFG, pooling_type=pooling).eval()
    ids = torch.randint(0, 30522, (1, 4))
    padded = torch.cat([ids, torch.randint(0, 30522, (1, 3))], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    assert torch.allclose(enc(ids), enc(padded, mask), atol=1e-5)


def test_max_pooling_takes_max_of_real_tokens_only():
    h = torch.tensor([[[1., 5.], [3., 2.], [9., 9.]]])  # last token is padding
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(max_pooling(h, mask), torch.tensor([[3., 5.]]))


def test_pool_without_mask_uses_all_tokens():
    enc = TransformerEncoder(**CFG, pooling_type="max")
    h = torch.randn(2, 5, 128)
    assert torch.equal(enc.pool(h), h.max(dim=1).values)


# --------------------------------------------------------------------------- #
# With BERT weights: same as HuggingFace
# --------------------------------------------------------------------------- #
def test_course_encoder_with_bert_weights_matches_huggingface():
    config = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64)
    bert = BertModel(config, add_pooling_layer=False).eval()
    enc = TransformerEncoder(n_layers=2, n_heads=4, emb_dim=32, hidden_dim=8, feed_forward_dim=64,
                             vocab_size=200, pooling_type="cls").eval()
    copy_bert_weights(enc.model, bert.state_dict())
    ids, mask = batch(vocab=200)
    with torch.no_grad():
        ref_cls = bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        assert torch.allclose(enc(ids, mask), ref_cls, atol=1e-5)
