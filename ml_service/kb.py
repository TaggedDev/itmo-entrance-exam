import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ml_service.config import Settings
from ml_service.embeddings import build_embeddings


def get_chroma(settings: Settings) -> Chroma:
    embeddings = build_embeddings(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        return Chroma(
            client=client,
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=str(settings.data_dir / "chroma_local"),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )


def reindex_knowledge(settings: Settings) -> tuple[int, int]:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = get_chroma(settings)
    try:
        vectorstore.delete_collection()
    except Exception:
        pass
    vectorstore = get_chroma(settings)

    documents: list[Document] = []
    files = sorted(settings.knowledge_dir.glob("*.txt"))
    for path in files:
        domain = path.stem
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(_split_text(text)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"domain": domain, "source": path.name, "chunk": index},
                )
            )
    if documents:
        vectorstore.add_documents(documents)
    return len(files), len(documents)


def retrieve_context(settings: Settings, query: str, k: int = 2) -> list[dict[str, object]]:
    vectorstore = get_chroma(settings)
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        return []
    contexts: list[dict[str, object]] = []
    for document, distance in results:
        score = 1.0 / (1.0 + max(float(distance), 0.0))
        contexts.append(
            {
                "domain": str(document.metadata.get("domain", "unknown")),
                "text": document.page_content,
                "score": round(float(score), 3),
            }
        )
    return contexts


def _split_text(text: str, max_chars: int = 700) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
