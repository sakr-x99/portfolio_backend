"""
Text Chunker
Splits Markdown knowledge files into semantic chunks for embedding and indexing.
Uses recursive character splitting with Markdown-aware separators.
"""
import logging
import os
from typing import List, Dict
from . import config

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_file(filepath: str) -> List[Dict]:
    """
    Read a single Markdown file, split it into chunks using LangChain splitters.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    source = filename.replace(".md", "")

    # 1. Split by Markdown headers to preserve context
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)

    # 2. Further split large sections into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(md_header_splits)

    result = []
    for i, split in enumerate(splits):
        # Extract heading from metadata if available
        heading = ""
        for h in ["Header 3", "Header 2", "Header 1"]:
            if h in split.metadata:
                heading = split.metadata[h]
                break

        result.append({
            "text": split.page_content,
            "metadata": {
                "source": source,
                "file": filename,
                "chunk_index": i,
                "heading": heading,
            }
        })

    return result


def chunk_all_knowledge() -> List[Dict]:
    """
    Process all Markdown files in the knowledge directory.
    Returns a flat list of all chunks with metadata.
    """
    all_chunks = []

    if not os.path.exists(config.KNOWLEDGE_DIR):
        logger.warning("Knowledge directory not found. Run extraction first.")
        return all_chunks

    for filename in sorted(os.listdir(config.KNOWLEDGE_DIR)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(config.KNOWLEDGE_DIR, filename)
        file_chunks = chunk_file(filepath)
        all_chunks.extend(file_chunks)
        logger.info("Chunked %s → %d chunks", filename, len(file_chunks))

    logger.info("Total: %d chunks ready for embedding", len(all_chunks))
    return all_chunks
