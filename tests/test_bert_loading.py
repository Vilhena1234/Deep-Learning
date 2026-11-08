"""
Test that copying BERT weights into our encoder reproduces HuggingFace's output.
Uses a small randomly initialised BERT, so it needs no download and runs in seconds.
(check_bert.py does the same with the real pretrained bert-base-uncased.)
"""

import os
import sys

import torch
from transformers import BertConfig, BertModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrievers.models.bert_loading import copy_bert_weights, encoder_from_bert_config  # noqa: E402

torch.manual_seed(0)


def small_bert():
    # "eager" = plain attention code; needed for HuggingFace to return attention maps
    config = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=3,
                        num_attention_heads=4, intermediate_size=64,
                        attn_implementation="eager")
    bert = BertModel(config, add_pooling_layer=False).eval()
    ours = encoder_from_bert_config(config)
    copy_bert_weights(ours, bert.state_dict())
    return bert, ours.eval()


def test_same_output_as_huggingface_bert():
    bert, ours = small_bert()
    ids = torch.randint(0, 200, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, 6:] = 0                                            # padding in sentence 2

    with torch.no_grad():
        ref = bert(input_ids=ids, attention_mask=mask).last_hidden_state
        out = ours(ids, mask)

    assert torch.allclose(out[0], ref[0], atol=1e-5)
    assert torch.allclose(out[1, :6], ref[1, :6], atol=1e-5)   # real tokens only


def test_same_attention_maps_as_huggingface_bert():
    bert, ours = small_bert()
    ids = torch.randint(0, 200, (1, 7))
    with torch.no_grad():
        ref = bert(input_ids=ids, output_attentions=True).attentions
        ours(ids)
    assert ref is not None and len(ref) == len(ours.attention_maps()) == 3   # really compared
    for ref_map, our_map in zip(ref, ours.attention_maps()):
        assert torch.allclose(our_map, ref_map, atol=1e-5)
