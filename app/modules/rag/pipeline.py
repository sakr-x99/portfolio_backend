"""
RAG Pipeline — Main Orchestrator
Connects all RAG components: extraction → chunking → embedding → indexing → retrieval → generation.
"""
import httpx
import os
from typing import List, Dict

from . import knowledge_extractor, chunker, embeddings, vector_store, prompt
from app.services.ai.manager import ai_manager


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
    print("\n═══ RAG Indexing Pipeline ═══\n")

    # Step 1: Extract knowledge from database
    print("Step 1/4: Extracting knowledge from database...")
    files = knowledge_extractor.extract_all()
    print(f"  → Generated {len(files)} knowledge files\n")

    # Step 2: Chunk the knowledge files
    print("Step 2/4: Chunking knowledge files...")
    chunks = chunker.chunk_all_knowledge()
    if not chunks:
        return {"status": "error", "message": "No chunks generated"}
    print(f"  → Created {len(chunks)} chunks\n")

    # Step 3: Generate embeddings
    print("Step 3/4: Generating embeddings...")
    texts = [c["text"] for c in chunks]
    chunk_embeddings = embeddings.embed_texts(texts)
    print(f"  → Generated {len(chunk_embeddings)} embeddings\n")

    # Step 4: Index into Qdrant
    print("Step 4/4: Indexing into Qdrant...")
    vector_store.clear_collection()
    vector_store.index_chunks(chunks, chunk_embeddings)

    info = vector_store.get_collection_info()
    print(f"\n═══ Indexing Complete ═══")
    print(f"  Points in Qdrant: {info.get('points_count', 'unknown')}\n")

    return {
        "status": "success",
        "files_generated": len(files),
        "chunks_created": len(chunks),
        "embeddings_generated": len(chunk_embeddings),
        "points_indexed": info.get("points_count", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve relevant context for a query:
    1. Generate query embedding
    2. Search Qdrant for similar chunks
    3. Return ranked results with scores
    """
    query_embedding = embeddings.embed_query(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

from .graph import rag_app

async def prepare_rag_state(
    question: str,
    conversation_history: List[Dict] = None,
) -> Dict:
    """
    Run all RAG pre-generation steps via the LangGraph nodes:
    1. Intent classification
    2. Query rewrite
    3. Context retrieval from Qdrant
    4. Lead capture (if hiring intent)

    Returns the full state ready for prompt building + streaming generation.
    This avoids duplicating graph logic in streaming endpoints.
    """
    history = list(conversation_history or [])

    state = {
        "question": question,
        "history": history,
        "context": [],
        "answer": "",
        "sources": [],
        "intent": "chat",
        "lead_data": {},
        "summary": ""
    }

    from .graph import classify_intent_node, rewrite_node, lead_capture_node

    intent_result = await classify_intent_node(state)
    state.update(intent_result)

    rewrite_result = await rewrite_node(state)
    state.update(rewrite_result)

    state["context"] = retrieve_context(state["question"], top_k=3)
    state["sources"] = list(set(c["source"] for c in state["context"]))

    if state["intent"] == "hiring":
        lead_result = await lead_capture_node(state)
        state.update(lead_result)

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
    # Build initial state
    history = list(conversation_history or [])
    # We don't append the question to history here because the graph handles it 
    # or the prompt builder handles it by combining context + history + current question.
    
    # Run graph
    inputs = {
        "question": question,
        "history": history,
        "context": [],
        "answer": "",
        "sources": [],
        "intent": "chat",
        "lead_data": {},
        "summary": ""
    }
    
    result = await rag_app.ainvoke(inputs)
    
    return {
        "content": result["answer"],
        "sources": result["sources"],
        "chunks_used": len(result["context"]),
        "intent": result["intent"],
        "summary": result["summary"]
    }
