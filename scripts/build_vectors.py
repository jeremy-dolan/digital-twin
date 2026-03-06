#!/usr/bin/env python3
"""
Rebuild the ChromaDB vector store from biography.txt.
Updated chromadb/ directory should committed afterwards.
Usage: python scripts/build_vectors.py
"""

import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import rag


def main():
    load_dotenv()

    print(f"Reading '{config.BIOGRAPHY_TXT.relative_to(config.BASE_DIR)}'")
    text = config.BIOGRAPHY_TXT.read_text()
    chunks = rag.chunk_curated_lines(text)
    for chunk in chunks:
        chunk.metadata["source"] = config.BIOGRAPHY_TXT.name
    sections = set(c.metadata['section'] for c in chunks)
    print(f"  Read {len(chunks)} chunks across {len(sections)} sections")

    print(f"Embedding chunks via OpenAI '{config.EMBEDDING_MODEL}'")
    oai_client = OpenAI()
    chunks = rag.embed_chunks(oai_client, chunks)
    embedded = sum(1 for c in chunks if c.embedding is not None)
    print(f"  Chunks embedded: {embedded}/{len(chunks)}")
    if chunks[0].embedding is not None:
        print(f"  Dimensions per vector: {len(chunks[0].embedding)}")
        print(f"  First vector begins: [{', '.join(f'{x:.6f}' for x in chunks[0].embedding[:4])}, ...]")

    print(f"Storing to ChromaDB at '{config.CHROMA_PATH.relative_to(config.BASE_DIR)}'")
    chroma_client = chromadb.PersistentClient(config.CHROMA_PATH, config.CHROMA_CLIENT_SETTINGS)
    collection = rag.db_store_embeds(chroma_client, config.CHROMA_COLLECTION_NAME, chunks)
    try:
        geometry = collection.configuration['hnsw']['space']           # type: ignore[index]
        emb_fn = collection.configuration['embedding_function'].name() # type: ignore[index]
    except (KeyError, TypeError):
        geometry = '<config lookup failed>'
        emb_fn = '<config lookup failed>'
    print(f"  Created collection: '{collection.name}'")
    print(f"  Distance measure: {geometry}")
    print(f'  Embedding function: {emb_fn}')
    print(f"  Total embeddings: {collection.count()}")
  
    print("Done.")


if __name__ == "__main__":
    main()
