import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ml_service.config import Settings
from ml_service.embeddings import build_embeddings
from ml_service.tracing import LangfuseTracer


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


def reindex_knowledge(settings: Settings, tracer: LangfuseTracer | None = None) -> tuple[int, int]:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = get_chroma(settings)
    with _observation(tracer, "chroma_delete_collection", input={"collection": settings.chroma_collection}):
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
    vectorstore = get_chroma(settings)

    documents: list[Document] = []
    files = sorted(settings.knowledge_dir.glob("*.txt"))
    with _observation(
        tracer,
        "knowledge_chunk_documents",
        input={"knowledge_dir": str(settings.knowledge_dir), "files": [path.name for path in files]},
    ) as observation:
        for path in files:
            domain = path.stem
            text = path.read_text(encoding="utf-8")
            chunks = _split_text(text)
            for index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"domain": domain, "source": path.name, "chunk": index},
                    )
                )
        observation.update(
            output={
                "indexed_files": len(files),
                "indexed_chunks": len(documents),
                "chunk_characters": [len(document.page_content) for document in documents],
            }
        )
    if documents:
        with _observation(
            tracer,
            "chroma_add_documents",
            as_type="embedding",
            input={
                "collection": settings.chroma_collection,
                "documents": len(documents),
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
            },
        ) as observation:
            ids = vectorstore.add_documents(documents)
            observation.update(output={"ids": ids, "documents": len(ids)})
    return len(files), len(documents)


def inspect_knowledge(settings: Settings, limit: int = 10) -> dict[str, object]:
    vectorstore = get_chroma(settings)
    collection = vectorstore._collection
    count = collection.count()
    result = collection.get(
        limit=limit,
        include=["documents", "embeddings", "metadatas"],
    )
    items: list[dict[str, object]] = []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    ids = result.get("ids") or []
    for index, document in enumerate(documents):
        embedding = embeddings[index] if embeddings is not None and index < len(embeddings) else []
        metadata = metadatas[index] if index < len(metadatas) else {}
        item_id = ids[index] if index < len(ids) else ""
        items.append(
            {
                "id": item_id,
                "metadata": metadata or {},
                "text": document,
                "characters": len(document),
                "embedding_dimensions": len(embedding),
                "embedding_preview": [round(float(value), 6) for value in embedding[:8]],
            }
        )
    return {
        "collection": settings.chroma_collection,
        "count": count,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "items": items,
    }


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
                "source": str(document.metadata.get("source", "unknown")),
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


def _observation(tracer: LangfuseTracer | None, name: str, **kwargs: object):
    if tracer is None:
        from contextlib import nullcontext

        return nullcontext(type("NoopObservation", (), {"update": lambda self, **_kwargs: None})())
    return tracer.observation(name, **kwargs)
