import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


from src.retrievers.models.transformer import TransformerEncoder as EncoderStack
from src.retrievers.models.pooling import cls_pooling, mean_pooling, max_pooling


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int,
                 vocab_size: int = 30522, pooling_type: str = 'cls', dropout: float = 0.1,
                 max_len: int = 512, positions: str = 'learned'):
        """
        n_layers:         number of encoder layers (N)
        n_heads:          number of attention heads (h)
        emb_dim:          size of each token vector (d_model)
        hidden_dim:       size of each attention head (d_k); must equal emb_dim / n_heads
                          (config/model/encoder.yaml: 128 / 4 = 32)
        feed_forward_dim: hidden size of the feed-forward network (d_ff)

        CHANGE: extra arguments with defaults, needed to build the model:
        vocab_size:       number of token ids; 30522 = bert-base-uncased tokenizer
        pooling_type:     'cls' | 'mean' | 'max'  (replaces self.cfg.model.pooling_type)
        dropout, max_len, positions ('learned' | 'sinusoidal' | 'none')
        """
        super().__init__()   # CHANGE: was missing; nn.Module must be initialised first
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.max_len = max_len
        self.positions = positions
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        assert self.emb_dim == self.n_heads * self.hidden_dim, (
            f"hidden_dim is the size of one attention head, so emb_dim must equal "
            f"n_heads * hidden_dim; got {self.emb_dim} != {self.n_heads} * {self.hidden_dim}")
        assert self.pooling_type in ('cls', 'mean', 'max'), f"unknown pooling_type {self.pooling_type}"

        # Embeddings (+ positions) -> n_layers x [attention -> Add & Norm -> FFN -> Add & Norm]
        self.model = EncoderStack(
            vocab_size=self.vocab_size,
            d_model=self.emb_dim,
            N=self.n_layers,
            h=self.n_heads,
            d_ff=self.feed_forward_dim,
            dropout=self.dropout,
            max_len=self.max_len,
            positions=self.positions,
        )

    def pool(self, x, attention_mask=None):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension
        Returns: A pooled vector of shape batch_size x emb_dim

        CHANGE: takes the attention mask (1 = real token, 0 = padding) so that mean and
        max pooling ignore padding; uses self.pooling_type instead of self.cfg.
        """
        if attention_mask is None:   # no padding: every token is real
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)

        if self.pooling_type == 'cls':
            return cls_pooling(x)

        elif self.pooling_type == 'max':
            return max_pooling(x, attention_mask)

        elif self.pooling_type == 'mean':
            return mean_pooling(x, attention_mask)

    def forward(self, x: torch.tensor, attention_mask: torch.tensor = None) -> torch.tensor:
        """
        Run a forward pass on the model

        x:              batch_size x sequence_length token ids (from the tokenizer)
        attention_mask: batch_size x sequence_length, 1 = real token, 0 = padding
        Returns:        batch_size x emb_dim, one vector per text (for DPR)
        """
        token_vectors = self.model(x, attention_mask)        # batch_size x seq_len x emb_dim
        return self.pool(token_vectors, attention_mask)

    def attention_maps(self):
        """
        Attention weights from the last forward pass: a list with one
        batch_size x n_heads x seq_len x seq_len tensor per layer (for the attention-map plots).
        """
        return self.model.attention_maps()

    @classmethod
    def from_config(cls, model_cfg, vocab_size: int = 30522):
        """Build from config/model/encoder.yaml, e.g. TransformerEncoder.from_config(cfg.model)."""
        return cls(n_layers=model_cfg.n_layers, n_heads=model_cfg.n_heads, emb_dim=model_cfg.emb_dim,
                   hidden_dim=model_cfg.hidden_dim, feed_forward_dim=model_cfg.feed_forward_dim,
                   vocab_size=vocab_size, pooling_type=model_cfg.pooling_type)
