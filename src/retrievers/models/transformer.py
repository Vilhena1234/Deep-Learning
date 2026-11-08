"""
Transformer encoder components (Person 3 — Gonçalo).

Built from the 02456 course material:
  * `attention`, `MultiHeadedAttention`, `PositionalEncoding`:
      Notebook 6.1 "Deep learning with Transformers" (cells IV.a and IV.b),
      itself based on "The Annotated Transformer" (Rush, 2018).
      Changes are marked with "CHANGE:" comments.
  * `LayerNorm`:  normalisation formula from Week 3, slide 22
      (mean 0, variance 1, then learned scale gamma and shift beta),
      applied over the features of each token instead of over the batch.
  * `SublayerConnection`: residual connection from Week 3, slide 21
      (h = h + f(h)), followed by LayerNorm ("Add & Norm", Notebook 6.1 IV.b).
  * `PositionwiseFeedForward`: two-layer feed-forward network (Weeks 1 and 3)
      applied to every token separately (Notebook 6.1 IV.b).

Shapes:  B = batch size, T = sequence length, d_model = token vector size,
         h = number of heads, d_k = d_model // h.
Padding mask: (B, T) tensor, 1/True = real token, 0/False = padding
              (same as HuggingFace `attention_mask`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Attention  (Notebook 6.1, IV.b — unchanged)
# --------------------------------------------------------------------------- #
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# --------------------------------------------------------------------------- #
# 2. Multi-head attention  (Notebook 6.1, IV.b)
# --------------------------------------------------------------------------- #
class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # linears[0..2] = query, key, value projections; linears[3] = output projection
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None  # store the attention maps (B, h, T, T) -> used for visualisation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            # CHANGE: accept a (B, T) padding mask and reshape it to (B, 1, 1, T)
            # so the same mask is applied to all h heads and all query positions.
            # (The notebook expects a (B, 1, T) mask and does mask.unsqueeze(1).)
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            else:
                mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# --------------------------------------------------------------------------- #
# 3. Layer normalisation  (Week 3, slide 22 — formula applied per token)
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """
    Same formula as batch normalisation on the slide:
        mu      = mean(z)
        sigma^2 = mean((z - mu)^2)
        z_norm  = (z - mu) / sqrt(sigma^2 + eps)
        out     = gamma * z_norm + beta
    but mu and sigma^2 are computed over the d_model features of EACH token,
    not over the batch. So it does not depend on other sentences in the batch
    or on sentence length, and behaves the same in training and testing.
    """

    def __init__(self, d_model, eps=1e-12):
        # eps = 1e-12 is the value BERT uses (PyTorch's default is 1e-5)
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z):
        mu = z.mean(dim=-1, keepdim=True)
        var = ((z - mu) ** 2).mean(dim=-1, keepdim=True)
        z_norm = (z - mu) / torch.sqrt(var + self.eps)
        return self.gamma * z_norm + self.beta


# --------------------------------------------------------------------------- #
# 4. Residual connection + Add & Norm  (Week 3 slide 21; Notebook 6.1 IV.b)
# --------------------------------------------------------------------------- #
class SublayerConnection(nn.Module):
    """
    "Add & Norm":  out = LayerNorm(x + Dropout(sublayer(x)))

    The "x +" is the residual (skip) connection from Week 3, slide 21:
    h^(l) = h^(l-1) + f(h^(l-1)). Its gradient is I + df/dh, so the gradient
    can always flow through the identity path, even in deep stacks.

    This is the post-norm order used in the original Transformer, in the
    lecture and in BERT.
    """

    def __init__(self, d_model, dropout, eps=1e-12):
        super().__init__()
        self.norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# --------------------------------------------------------------------------- #
# 5. Position-wise feed-forward network  (Weeks 1 and 3; Notebook 6.1 IV.b)
# --------------------------------------------------------------------------- #
class PositionwiseFeedForward(nn.Module):
    """
    A two-layer feed-forward network applied to every token separately:
        FFN(x) = W2 * activation(W1 * x + b1) + b2
    d_ff is usually 4 * d_model. The original Transformer uses ReLU;
    BERT uses GELU (a smooth version of ReLU).
    """

    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# --------------------------------------------------------------------------- #
# 6. Positional representations
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positions (Notebook 6.1, IV.a — unchanged except that
    adding to the input and dropout are moved to `Embeddings`).
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, T):
        return self.pe[:, :T]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positions: one trainable vector per position (as in BERT)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, T):
        return self.pe(torch.arange(T, device=self.pe.weight.device))[None]


class Embeddings(nn.Module):
    """
    Token ids -> vectors:  LayerNorm(token_embedding + position) then dropout.
    (Notebook 6.1 I.b for token embeddings; the LayerNorm on the embeddings is
    what BERT does.)
    """

    def __init__(self, vocab_size, d_model, max_len=512, positions="learned", dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        if positions == "sinusoidal":
            self.pos = PositionalEncoding(d_model, max_len)
        elif positions == "learned":
            self.pos = LearnedPositionalEmbedding(d_model, max_len)
        else:  # "none": used to show that attention alone ignores word order
            self.pos = None
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.tok(input_ids)
        if self.pos is not None:
            x = x + self.pos(input_ids.size(1))
        return self.dropout(self.norm(x))


# --------------------------------------------------------------------------- #
# 7. Encoder layer and full encoder
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    """
    One Transformer encoder layer (post-norm, as in the lecture and BERT):
        x = LayerNorm(x + Dropout(MultiHeadAttention(x)))
        x = LayerNorm(x + Dropout(FeedForward(x)))
    """

    def __init__(self, d_model, h, d_ff, dropout=0.1, activation="gelu", eps=1e-12):
        super().__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout, eps) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    """
    Token ids -> one vector per token.  Embeddings, then N encoder layers.
    `attention_maps()` returns the attention weights of every layer after a
    forward pass: a list of N tensors of shape (B, h, T, T).
    """

    def __init__(self, vocab_size, d_model=256, N=4, h=4, d_ff=1024, dropout=0.1,
                 max_len=512, positions="learned", activation="gelu"):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, max_len, positions, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout, activation) for _ in range(N)]
        )
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x  # (B, T, d_model)

    def attention_maps(self):
        return [layer.self_attn.attn for layer in self.layers]
