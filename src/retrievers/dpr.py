from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(...)
        document_encoder = TransformerEncoder(...)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)

    

        

