"""
Pooling (Person 3 — Gonçalo): one vector per token  ->  one vector per text.

The encoder returns (B, T, d_model). Retrieval needs one vector per question or
passage, (B, d_model). Two options, compared in the experiments:

  * CLS pooling:  take the vector of the first token ([CLS] in BERT).
                  Notebook 6.1, V.d uses DPR's `pooler_output`, which is built
                  from this token.
  * Mean pooling: average the vectors of all real tokens, ignoring padding.
  * Max pooling:  for each feature, keep the largest value over all real tokens
                  (asked for in the course starter code, config/model/encoder.yaml).
"""

import torch


def cls_pooling(token_vectors: torch.Tensor, attention_mask=None) -> torch.Tensor:
    """token_vectors: (B, T, d) -> (B, d). The mask is not needed (position 0 is always real)."""
    return token_vectors[:, 0]


def mean_pooling(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_vectors:  (B, T, d)
    attention_mask: (B, T), 1/True = real token, 0/False = padding
    Returns (B, d): sum of real token vectors / number of real tokens.
    """
    mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)   # (B, T, 1)
    summed = (token_vectors * mask).sum(dim=1)                     # (B, d)
    counts = mask.sum(dim=1).clamp(min=1e-9)                       # (B, 1), avoid division by 0
    return summed / counts


def max_pooling(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_vectors:  (B, T, d)
    attention_mask: (B, T), 1/True = real token, 0/False = padding
    Returns (B, d): for each of the d features, the largest value over the real tokens.
    Padding is set to -inf first so it can never be the maximum.
    """
    mask = attention_mask.bool().unsqueeze(-1)                    # (B, T, 1)
    masked = token_vectors.masked_fill(~mask, float("-inf"))      # padding -> -inf
    return masked.max(dim=1).values                               # (B, d)


POOLING = {"cls": cls_pooling, "mean": mean_pooling, "max": max_pooling}


def pool(token_vectors, attention_mask, method: str = "mean"):
    """Convenience wrapper: pool(h, mask, "cls"), pool(h, mask, "mean") or pool(h, mask, "max")."""
    return POOLING[method](token_vectors, attention_mask)
