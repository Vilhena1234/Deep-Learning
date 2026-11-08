"""
Tests for transformer.py and pooling.py.
Every component is checked against PyTorch's own implementation.
Run from the project folder with:  python -m pytest tests -v
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from src.retrievers.models.pooling import cls_pooling, mean_pooling  # noqa: E402
from src.retrievers.models.transformer import (  # noqa: E402
    EncoderLayer, LayerNorm, MultiHeadedAttention, PositionalEncoding,
    PositionwiseFeedForward, SublayerConnection, TransformerEncoder, attention,
)

torch.manual_seed(0)
TOL = 1e-5


def padding_mask(B=2, T=6):
    """Second sentence has its last 2 tokens as padding."""
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, -2:] = False
    return mask


# --------------------------------------------------------------------------- #
# 1. Attention
# --------------------------------------------------------------------------- #
def test_attention_matches_pytorch():
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))     # (B, h, T, d_k)
    ours, _ = attention(q, k, v)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=TOL)


def test_attention_with_mask_matches_pytorch_and_ignores_padding():
    B, h, T, d = 2, 3, 6, 8
    q, k, v = (torch.randn(B, h, T, d) for _ in range(3))
    mask = padding_mask(B, T)[:, None, None, :]
    ours, w = attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(ours, ref, atol=TOL)
    assert torch.allclose(w.sum(-1), torch.ones(B, h, T), atol=TOL)   # rows sum to 1
    assert torch.all(w[1, :, :, -2:] == 0)                            # no weight on padding


# --------------------------------------------------------------------------- #
# 2. Multi-head attention
# --------------------------------------------------------------------------- #
def copy_mha(ours: MultiHeadedAttention, ref: nn.MultiheadAttention):
    """Give our module exactly the same weights as PyTorch's."""
    d = ref.embed_dim
    with torch.no_grad():
        for i in range(3):                                    # q, k, v stacked in in_proj
            ours.linears[i].weight.copy_(ref.in_proj_weight[i * d:(i + 1) * d])
            ours.linears[i].bias.copy_(ref.in_proj_bias[i * d:(i + 1) * d])
        ours.linears[3].weight.copy_(ref.out_proj.weight)
        ours.linears[3].bias.copy_(ref.out_proj.bias)


def test_multihead_matches_pytorch():
    B, T, d, h = 2, 6, 16, 4
    x, mask = torch.randn(B, T, d), padding_mask(B, T)
    ref = nn.MultiheadAttention(d, h, batch_first=True).eval()
    ours = MultiHeadedAttention(h, d).eval()
    copy_mha(ours, ref)

    out = ours(x, x, x, mask)
    # PyTorch's key_padding_mask uses the opposite convention: True = ignore
    ref_out, ref_w = ref(x, x, x, key_padding_mask=~mask, average_attn_weights=False)
    assert ours.attn.shape == (B, h, T, T)
    assert torch.allclose(out[0], ref_out[0], atol=TOL)
    assert torch.allclose(out[1, :-2], ref_out[1, :-2], atol=TOL)   # real tokens only
    assert torch.allclose(ours.attn[1, :, :-2], ref_w[1, :, :-2], atol=TOL)


def test_padding_does_not_change_real_tokens():
    """Same sentence with and without extra padding must give the same output."""
    d, h = 16, 4
    mha = MultiHeadedAttention(h, d).eval()
    x = torch.randn(1, 4, d)
    x_pad = torch.cat([x, torch.randn(1, 3, d)], dim=1)      # 3 junk tokens
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    assert torch.allclose(mha(x, x, x), mha(x_pad, x_pad, x_pad, mask)[:, :4], atol=TOL)


# --------------------------------------------------------------------------- #
# 3. Layer norm, residual, feed-forward
# --------------------------------------------------------------------------- #
def test_layernorm_matches_pytorch():
    x = torch.randn(2, 6, 16) * 3 + 5
    ours, ref = LayerNorm(16, eps=1e-5), nn.LayerNorm(16, eps=1e-5)
    with torch.no_grad():
        ours.gamma.copy_(torch.randn(16)); ours.beta.copy_(torch.randn(16))
        ref.weight.copy_(ours.gamma); ref.bias.copy_(ours.beta)
    assert torch.allclose(ours(x), ref(x), atol=TOL)


def test_layernorm_gives_mean_0_variance_1_per_token():
    z = LayerNorm(16)(torch.randn(2, 6, 16) * 10 + 3)        # gamma=1, beta=0
    assert torch.allclose(z.mean(-1), torch.zeros(2, 6), atol=1e-5)
    assert torch.allclose(z.var(-1, unbiased=False), torch.ones(2, 6), atol=1e-4)


def test_layernorm_does_not_depend_on_other_sentences():
    """Unlike batch norm, a sentence's output does not change with the batch."""
    ln = LayerNorm(16)
    a = torch.randn(1, 6, 16)
    batch = torch.cat([a, torch.randn(3, 6, 16) * 100])
    assert torch.allclose(ln(a), ln(batch)[:1], atol=TOL)


def test_residual_gradient_has_identity_path():
    """Week 3 slide 21: d(x + f(x))/dx = I + df/dx. With f = 0, the gradient is exactly I."""
    x = torch.randn(3, 16, requires_grad=True)
    out = x + 0 * nn.Linear(16, 16)(x)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_feedforward_acts_on_each_token_separately():
    ffn = PositionwiseFeedForward(16, 64, dropout=0.0).eval()
    x = torch.randn(1, 5, 16)
    full = ffn(x)
    for t in range(5):
        assert torch.allclose(full[:, t], ffn(x[:, t:t + 1])[:, 0], atol=TOL)


# --------------------------------------------------------------------------- #
# 4. Full encoder layer
# --------------------------------------------------------------------------- #
def copy_layer(ours: EncoderLayer, ref: nn.TransformerEncoderLayer):
    copy_mha(ours.self_attn, ref.self_attn)
    with torch.no_grad():
        ours.feed_forward.w_1.load_state_dict(ref.linear1.state_dict())
        ours.feed_forward.w_2.load_state_dict(ref.linear2.state_dict())
        for our_norm, ref_norm in [(ours.sublayer[0].norm, ref.norm1), (ours.sublayer[1].norm, ref.norm2)]:
            our_norm.gamma.copy_(ref_norm.weight); our_norm.beta.copy_(ref_norm.bias)


def test_encoder_layer_matches_pytorch():
    B, T, d, h, d_ff = 2, 6, 16, 4, 64
    x, mask = torch.randn(B, T, d), padding_mask(B, T)
    ref = nn.TransformerEncoderLayer(d, h, d_ff, dropout=0.0, activation="gelu",
                                     batch_first=True, norm_first=False, layer_norm_eps=1e-5).eval()
    ours = EncoderLayer(d, h, d_ff, dropout=0.0, activation="gelu", eps=1e-5).eval()
    copy_layer(ours, ref)

    out = ours(x, mask)
    ref_out = ref(x, src_key_padding_mask=~mask)
    assert torch.allclose(out[0], ref_out[0], atol=1e-4)
    assert torch.allclose(out[1, :-2], ref_out[1, :-2], atol=1e-4)


# --------------------------------------------------------------------------- #
# 5. Positions: why they are needed (Week 8 idea: permutation equivariance)
# --------------------------------------------------------------------------- #
def test_without_positions_word_order_is_ignored():
    """Shuffling the tokens only shuffles the outputs: the model cannot see order."""
    enc = TransformerEncoder(100, d_model=32, N=2, h=4, d_ff=64, dropout=0.0, positions="none").eval()
    ids = torch.tensor([[5, 17, 42, 8, 99]])
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(enc(ids)[:, perm], enc(ids[:, perm]), atol=1e-5)


def test_with_positions_word_order_matters():
    enc = TransformerEncoder(100, d_model=32, N=2, h=4, d_ff=64, dropout=0.0, positions="sinusoidal").eval()
    ids = torch.tensor([[5, 17, 42, 8, 99]])
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert not torch.allclose(enc(ids)[:, perm], enc(ids[:, perm]), atol=1e-3)


def test_sinusoidal_positions_are_unique_and_bounded():
    pe = PositionalEncoding(64, max_len=100)(100)[0]           # (100, 64)
    assert pe.abs().max() <= 1.0
    assert torch.unique(pe, dim=0).shape[0] == 100              # every position different


def test_encoder_shapes_and_attention_maps():
    enc = TransformerEncoder(100, d_model=32, N=3, h=4, d_ff=64).eval()
    ids, mask = torch.randint(0, 100, (2, 7)), padding_mask(2, 7)
    out = enc(ids, mask)
    maps = enc.attention_maps()
    assert out.shape == (2, 7, 32)
    assert len(maps) == 3 and maps[0].shape == (2, 4, 7, 7)


# --------------------------------------------------------------------------- #
# 6. Pooling
# --------------------------------------------------------------------------- #
def test_cls_pooling_takes_first_token():
    h = torch.randn(2, 6, 16)
    assert torch.equal(cls_pooling(h), h[:, 0])


def test_mean_pooling_ignores_padding():
    h, mask = torch.randn(2, 6, 16), padding_mask(2, 6)
    pooled = mean_pooling(h, mask)
    assert torch.allclose(pooled[0], h[0].mean(0), atol=TOL)
    assert torch.allclose(pooled[1], h[1, :4].mean(0), atol=TOL)  # last 2 are padding
