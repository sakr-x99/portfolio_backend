"""
RAG Pipeline — Main Orchestrator
Connects all RAG components: extraction → chunking → embedding → indexing → retrieval → generation.
"""
import logging
from typing import List, Dict

from . import knowledge_extractor, chunker, embeddings, vector_store

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEXING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def index_knowledge() -> Dict:
    """
    Full indexing pipeline:
    1. Extract data from PostgreSQL → Markdown files
    2. Chunk Markdown files into semantic segments
    3. Generate embeddings for each chunk
    4. Index chunks + embeddings into Qdrant

    Returns status dict with stats.
    """
    logger.info("═══ RAG Indexing Pipeline ═══")

    logger.info("Step 1/4: Extracting knowledge from database...")
    files = knowledge_extractor.extract_all()
    logger.info("→ Generated %d knowledge files", len(files))

    logger.info("Step 2/4: Chunking knowledge files...")
    chunks = chunker.chunk_all_knowledge()
    if not chunks:
        return {"status": "error", "message": "No chunks generated"}
    logger.info("→ Created %d chunks", len(chunks))

    logger.info("Step 3/4: Generating embeddings...")
    texts = [c["text"] for c in chunks]
    chunk_embeddings = embeddings.embed_texts(texts)
    logger.info("→ Generated %d embeddings", len(chunk_embeddings))

    logger.info("Step 4/4: Indexing into Qdrant...")
    vector_store.clear_collection()
    vector_store.index_chunks(chunks, chunk_embeddings)

    info = vector_store.get_collection_info()
    points_count = info.get("points_count", 0)
    logger.info("═══ Indexing Complete ═══")
    logger.info("Points in Qdrant: %s", points_count)

    return {
        "status": "success",
        "files_generated": len(files),
        "chunks_created": len(chunks),
        "embeddings_generated": len(chunk_embeddings),
        "points_indexed": points_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve relevant context for a query:
    1. Generate query embedding
    2. Search Qdrant for similar chunks
    3. Return ranked results with scores
    """
    from .config import TOP_K as DEFAULT_TOP_K
    top_k = top_k or DEFAULT_TOP_K
    query_embedding = embeddings.embed_query(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def prepare_rag_state(
    question: str,
    conversation_history: List[Dict] = None,
) -> Dict:
    """
    Run all RAG pre-generation steps via the shared LangGraph compiled graph.
    Uses the same graph as generate_answer to avoid duplicate logic.

    Returns the full state ready for prompt building + streaming generation.
    """
    history = list(conversation_history or [])

    inputs = {
        "question": question,
        "original_question": question,
        "history": history,
        "context": [],
        "answer": "",
        "sources": [],
        "intent": "chat",
        "lead_data": {},
        "summary": "",
    }

    from .graph import get_rag_app
    graph = get_rag_app()

    # Run first part of graph: classify → rewrite → retrieve → lead_capture
    # We use async generator iteration to stop after lead_capture
    state = inputs
    async for event in graph.astream(inputs):
        # The last event is the final state — we take all keys up to generate
        for node_name, node_output in event.items():
            if node_name == "generate":
                break
            if isinstance(node_output, dict):
                state.update(node_output)

    return state


async def generate_answer(
    question: str,
    conversation_history: List[Dict] = None,
) -> Dict:
    """
    LangGraph-powered RAG pipeline:
    1. Invoke the graph (Retrieve -> Generate)
    2. Return the final answer with sources
    """
    history = list(conversation_history or [])

    inputs = {
        "question": question,
        "original_question": question,
        "history": history,
        "context": [],
        "answer": "",
        "sources": [],
        "intent": "chat",
        "lead_data": {},
        "summary": "",
    }

    from .graph import get_rag_app
    result = await get_rag_app().ainvoke(inputs)

    return {
        "content": result["answer"],
        "sources": result["sources"],
        "chunks_used": len(result["context"]),
        "intent": result["intent"],
        "summary": result["summary"],
    }
