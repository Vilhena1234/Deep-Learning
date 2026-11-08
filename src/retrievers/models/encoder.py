import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.model = None
        raise NotImplementedError()

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'max':
            raise NotImplementedError()
        
        elif self.cfg.model.pooling_type == 'mean':
            raise NotImplementedError()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        raise NotImplementedError()
