"""Cross-encoder reranking of first-stage candidates."""
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device="cpu")

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self.model.predict(pairs, batch_size=16)]
