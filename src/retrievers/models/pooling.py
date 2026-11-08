

import torch


def cls_pooling(token_vectors: torch.Tensor, attention_mask=None) -> torch.Tensor:
 
    return token_vectors[:, 0]


def mean_pooling(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
   
    mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)   # (B, T, 1)
    summed = (token_vectors * mask).sum(dim=1)                     # (B, d)
    counts = mask.sum(dim=1).clamp(min=1e-9)                       # (B, 1), avoid division by 0
    return summed / counts


def max_pooling(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    
    mask = attention_mask.bool().unsqueeze(-1)                    # (B, T, 1)
    masked = token_vectors.masked_fill(~mask, float("-inf"))      # padding -> -inf
    return masked.max(dim=1).values                               # (B, d)


POOLING = {"cls": cls_pooling, "mean": mean_pooling, "max": max_pooling}


def pool(token_vectors, attention_mask, method: str = "mean"):
    
    return POOLING[method](token_vectors, attention_mask)
