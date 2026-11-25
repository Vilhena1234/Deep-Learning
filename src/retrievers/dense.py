from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

from src.retrievers.base_retriever import Retriever
from src.retrievers.models.pooling import mean_pooling


class DenseRetriever(Retriever):
    def __init__(self, documents, model_path, batch_size=32, max_length=384, vectors_path=None):
        super().__init__(documents)
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        if vectors_path is not None and Path(vectors_path).exists():
            self.LoadVectors(vectors_path)
        else:
            texts = [doc.GetText() for doc in self.index.GetDocuments()]
            self.doc_vectors = self.Encode(texts)
            if vectors_path is not None:
                self.SaveVectors(vectors_path)

    def DocumentIds(self):
        return [doc.GetId() for doc in self.index.GetDocuments()]

    def SaveVectors(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": self.DocumentIds(), "model_path": self.model_path,
                    "vectors": self.doc_vectors.cpu()}, path)

    def LoadVectors(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["ids"] != self.DocumentIds():
            raise ValueError("saved vectors belong to different documents")
        if saved["model_path"] != self.model_path:
            raise ValueError("saved vectors were made by a different model")
        self.doc_vectors = saved["vectors"]


    @torch.no_grad()
    def Encode(self, texts):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = [texts[i] for i in order[start:start + self.batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            tokens = self.model(**inputs).last_hidden_state
            chunks.append(mean_pooling(tokens, inputs["attention_mask"]))
        sorted_vectors = torch.cat(chunks)
        vectors = torch.empty_like(sorted_vectors)
        vectors[order] = sorted_vectors
        return vectors

    def CalculateScores(self, queries):
        query_vectors = self.Encode(queries)
        scores = query_vectors @ self.doc_vectors.T
        return scores.cpu().numpy()

    def TopK(self, queries, k):
        scores = torch.from_numpy(self.CalculateScores(queries))
        values, positions = scores.topk(min(k, scores.shape[1]), dim=1)
        return [list(zip(p, v)) for p, v in zip(positions.tolist(), values.tolist())]
