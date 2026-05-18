"""
Vector Store — Qdrant Integration
Handles collection management, document indexing, and semantic similarity search.
"""
from typing import List, Dict, Optional
from . import config

# Lazy-init client
_client = None


def _get_qdrant():
    """Lazy-import qdrant_client models."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    return QdrantClient, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


def _get_client():
    """Get or create the Qdrant client (supports both local and cloud)."""
    global _client
    if _client is None:
        QdrantClient = _get_qdrant()[0]
        if config.QDRANT_API_KEY:
            _client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=30
            )
            print(f"✓ Qdrant Cloud connected: {config.QDRANT_URL}")
        else:
            _client = QdrantClient(url=config.QDRANT_URL, timeout=30)
            print(f"✓ Qdrant Local connected: {config.QDRANT_URL}")
    return _client


def ensure_collection():
    """
    Create the collection if it doesn't exist.
    Recreates it if dimensions mismatch (e.g. model change).
    """
    _, Distance, VectorParams, *_ = _get_qdrant()
    client = _get_client()
    collection_name = config.QDRANT_COLLECTION

    try:
        info = client.get_collection(collection_name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != config.EMBEDDING_DIMENSIONS:
            print(f"  ⚠ Dimension mismatch ({existing_dim} vs {config.EMBEDDING_DIMENSIONS}). Recreating...")
            client.delete_collection(collection_name)
            raise Exception("Force recreate")
        print(f"  ✓ Collection '{collection_name}' exists ({info.points_count} points)")
    except Exception:
        print(f"  ⏳ Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        print(f"  ✓ Collection '{collection_name}' created")


def index_chunks(chunks: List[Dict], embeddings: List[List[float]]):
    """
    Upsert chunks with their embeddings into Qdrant.
    """
    _, _, _, PointStruct, *_ = _get_qdrant()
    client = _get_client()
    collection_name = config.QDRANT_COLLECTION

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "file": chunk["metadata"]["file"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "heading": chunk["metadata"]["heading"],
                },
            )
        )

    batch_size = 100
    for batch_start in range(0, len(points), batch_size):
        batch = points[batch_start:batch_start + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"  ✓ Indexed {len(points)} chunks into Qdrant")


def search(
    query_embedding: List[float],
    top_k: int = None,
    score_threshold: float = None,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Perform semantic similarity search in Qdrant.
    """
    _, _, _, _, Filter, FieldCondition, MatchValue = _get_qdrant()
    client = _get_client()
    top_k = top_k or config.TOP_K
    score_threshold = score_threshold or config.SCORE_THRESHOLD

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k,
        score_threshold=score_threshold,
    )

    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "heading": hit.payload.get("heading", ""),
            "score": round(hit.score, 4),
        }
        for hit in results
    ]


def get_collection_info() -> Dict:
    """Get collection stats for health check."""
    try:
        client = _get_client()
        info = client.get_collection(config.QDRANT_COLLECTION)
        return {
            "status": "healthy",
            "collection": config.QDRANT_COLLECTION,
            "points_count": info.points_count,
            "vectors_size": info.config.params.vectors.size,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def clear_collection():
    """Delete and recreate the collection (for re-indexing)."""
    client = _get_client()
    try:
        client.delete_collection(config.QDRANT_COLLECTION)
        print(f"  ✓ Deleted collection '{config.QDRANT_COLLECTION}'")
    except Exception:
        pass
    ensure_collection()
