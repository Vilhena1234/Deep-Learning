"""
Load pretrained BERT weights into our own TransformerEncoder (Person 3 — Gonçalo).

Why: if our encoder, with BERT's weights copied in, gives the same output as
HuggingFace's BERT, our implementation of every component is correct on a real
model, not only on random test numbers.

How: BERT's weights are stored under names like
    encoder.layer.3.attention.self.query.weight
We copy each one into the matching part of our model:
    layers[3].self_attn.linears[0].weight

Usage:
    from bert_loading import load_bert
    encoder, tokenizer = load_bert("bert-base-uncased")
    batch = tokenizer(["what is a dividend?"], return_tensors="pt", padding=True)
    h = encoder(batch["input_ids"], batch["attention_mask"])     # (B, T, 768)
"""

import torch

from src.retrievers.models.transformer import TransformerEncoder


def encoder_from_bert_config(config) -> TransformerEncoder:
    """Build our encoder with the same sizes as a HuggingFace BertConfig."""
    assert config.hidden_act == "gelu", "our feed-forward supports exact GELU (as in BERT)"
    model = TransformerEncoder(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,               # 768 in bert-base
        N=config.num_hidden_layers,               # 12
        h=config.num_attention_heads,             # 12
        d_ff=config.intermediate_size,            # 3072
        dropout=config.hidden_dropout_prob,
        max_len=config.max_position_embeddings,   # 512
        positions="learned",                      # BERT learns its position vectors
        activation="gelu",
    )
    # BERT uses eps = 1e-12 in every LayerNorm (our default too); set it explicitly anyway
    for m in model.modules():
        if hasattr(m, "eps"):
            m.eps = config.layer_norm_eps
    return model


@torch.no_grad()
def copy_bert_weights(ours: TransformerEncoder, bert_state: dict) -> None:
    """
    Copy every weight from a HuggingFace BertModel state_dict into our encoder.
    Raises an error if a shape does not match, so mistakes cannot go unnoticed.
    """

    def put(target: torch.Tensor, name: str, value=None):
        value = bert_state[name] if value is None else value
        assert target.shape == value.shape, f"{name}: ours {tuple(target.shape)} vs BERT {tuple(value.shape)}"
        target.copy_(value)

    # ---- Embeddings ------------------------------------------------------- #
    e = ours.embeddings
    put(e.tok.weight, "embeddings.word_embeddings.weight")

    # BERT adds a third vector to each token: the "token type" (sentence A or B).
    # We always encode one text at a time, so the type is always 0. Adding the
    # same type-0 vector to every token is the same as adding it to every
    # position vector, so we fold it in here instead of adding a new layer.
    pos = bert_state["embeddings.position_embeddings.weight"]
    type0 = bert_state["embeddings.token_type_embeddings.weight"][0]
    put(e.pos.pe.weight, "embeddings.position_embeddings.weight", pos + type0)

    put(e.norm.gamma, "embeddings.LayerNorm.weight")
    put(e.norm.beta, "embeddings.LayerNorm.bias")

    # ---- Encoder layers ---------------------------------------------------- #
    for i, layer in enumerate(ours.layers):
        p = f"encoder.layer.{i}."
        attn = layer.self_attn.linears
        # step 2: query, key, value and output projections
        for j, part in enumerate(["attention.self.query", "attention.self.key",
                                  "attention.self.value", "attention.output.dense"]):
            put(attn[j].weight, p + part + ".weight")
            put(attn[j].bias, p + part + ".bias")
        # step 5: Add & Norm after attention
        put(layer.sublayer[0].norm.gamma, p + "attention.output.LayerNorm.weight")
        put(layer.sublayer[0].norm.beta, p + "attention.output.LayerNorm.bias")
        # step 6: feed-forward (768 -> 3072 -> 768)
        put(layer.feed_forward.w_1.weight, p + "intermediate.dense.weight")
        put(layer.feed_forward.w_1.bias, p + "intermediate.dense.bias")
        put(layer.feed_forward.w_2.weight, p + "output.dense.weight")
        put(layer.feed_forward.w_2.bias, p + "output.dense.bias")
        # step 5: Add & Norm after feed-forward
        put(layer.sublayer[1].norm.gamma, p + "output.LayerNorm.weight")
        put(layer.sublayer[1].norm.beta, p + "output.LayerNorm.bias")


def load_bert_encoder(model_name: str = "bert-base-uncased", pooling_type: str = "cls"):
    """
    Same as load_bert, but returns the course's TransformerEncoder (encoder.py), which
    pools to one vector per text. Returns (encoder, tokenizer).
    """
    from transformers import AutoTokenizer, BertModel
    from src.retrievers.models.encoder import TransformerEncoder as CourseEncoder

    bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
    c = bert.config
    encoder = CourseEncoder(n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                            emb_dim=c.hidden_size, hidden_dim=c.hidden_size // c.num_attention_heads,
                            feed_forward_dim=c.intermediate_size, vocab_size=c.vocab_size,
                            pooling_type=pooling_type, dropout=c.hidden_dropout_prob,
                            max_len=c.max_position_embeddings, positions="learned")
    for m in encoder.modules():
        if hasattr(m, "eps"):
            m.eps = c.layer_norm_eps
    copy_bert_weights(encoder.model, bert.state_dict())
    return encoder.eval(), AutoTokenizer.from_pretrained(model_name)


def load_bert(model_name: str = "bert-base-uncased"):
    """Download BERT from HuggingFace and return (our encoder with its weights, tokenizer)."""
    from transformers import AutoTokenizer, BertModel

    bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
    ours = encoder_from_bert_config(bert.config)
    copy_bert_weights(ours, bert.state_dict())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return ours.eval(), tokenizer
