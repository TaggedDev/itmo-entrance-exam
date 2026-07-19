import hashlib
import math
import re
from functools import cached_property

from langchain_core.embeddings import Embeddings


class DeterministicHashEmbeddings(Embeddings):
    """Small local embedding for tests and fully offline smoke checks."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\wа-яА-ЯёЁ]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class MultilingualE5Embeddings(Embeddings):
    """LangChain-compatible wrapper for local intfloat multilingual E5 models."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [self._ensure_prefix(text, "passage") for text in texts]
        return self._encode(passages)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._ensure_prefix(text, "query")])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _ensure_prefix(self, text: str, prefix: str) -> str:
        stripped = text.strip()
        if stripped.startswith(("query: ", "passage: ")):
            return stripped
        return f"{prefix}: {stripped}"


def build_embeddings(
    provider: str,
    model_name: str,
    device: str,
) -> Embeddings:
    if provider == "hash":
        return DeterministicHashEmbeddings()
    if provider == "e5":
        return MultilingualE5Embeddings(model_name=model_name, device=device)
    raise ValueError(f"Unsupported embedding provider: {provider}")
