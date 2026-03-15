import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

"""
# RAG pipeline:

## Indexing
1. **Load/clean/chunk data** — split source documents into ChunkedText objects
    - data was generated for purpose by an extended interview with Claude
    - since our data is highly curated we can just chunk by line
2. **Embed** — pass chunks to bi-encoder and obtain dense vectors
    - model: OpenAI `text-embedding-3-large` (3072 dimensions)
3. **Store** — save chunks plus embeddings/metadata for semantic search
    - chromadb collection: `bio_facts_large_embed`

## Query-time (R/A/G)
4. **Retrieve** — match user query to stored embeddings
    - get `k` most similar chunks
    - check for matches better than `config.DISTANCE_THRESHOLD`
    - return up to j matches within `gap`
5. **Augment** — add chosen chunks as context to the prompt
    - `build_context_injection()`
6. **Generate** — pass original user query with augmented prompt to LLM
"""

@dataclass
class ChunkedText:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | np.ndarray | None = None
    id: str | None = None
# class ChunkedText:
#     def __init__(self, text: str, metadata=None):
#         self.text = text
#         self.metadata = {} if metadata is None else metadata
#     def __repr__(self) -> str:
#         return f'ChunkedText(metadata={str(self.metadata)}, text="{self.text}"'


### Indexing: load, chunk, embed, store

# new per-line chunker for curated data. See -arch4 for old sentence-based chunk_text()
def chunk_curated_lines(text: str) -> list[ChunkedText]:
    """Split text into chunks by line, tracking '# section' headers as metadata."""
    chunks = []
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
        else:
            chunks.append(
                ChunkedText(text=stripped, metadata={
                    'section': section,
                    'chunk': len(chunks)+1,
                })
            )
    return chunks


def embed_strings(oai_client: OpenAI, strings: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using the OpenAI embeddings API.
    Sends all chunks in a single batched request (up to 2048 inputs).
    Returns a list of (normalized) vectors in the same order as the input.
    """
    response = oai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=strings
    )
    return [item.embedding for item in response.data]


def embed_chunks(oai_client: OpenAI, chunks: list[ChunkedText]) -> list[ChunkedText]:
    """Generate an embedding for each ChunkedText and attach it. Returns the same list."""
    embeddings = embed_strings(oai_client, [chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.metadata["embedding_model"] = config.EMBEDDING_MODEL
    return chunks


def db_store_embeds(
        chroma_client: chromadb.ClientAPI, # type: ignore[reportPrivateImportUsage]
        collection_name: str,
        chunks: list[ChunkedText], # broaden if adding another Chunk type
    ) -> chromadb.Collection:
    """Store embeddings from a list of ChunkedTexts as a new ChromaDB collection.
    (Deletes the collection first if it already exists.)"""
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)
    collection = chroma_client.create_collection(
        name=collection_name,
        configuration=config.CHROMA_COLLECTION_CONFIG,
    )

    # unzip chunks into parallel lists of texts/embeddings/metadatas, and generate unique IDs
    docs, embeds, metadata, ids = [], [], [], []
    for chunk in chunks:
        docs.append(chunk.text)
        embeds.append(chunk.embedding)
        metadata.append(chunk.metadata)
        ids.append(str(uuid.uuid4()))

    collection.add(ids=ids, embeddings=embeds, metadatas=metadata, documents=docs)
    return collection


def db_load_embeds(
        chroma_client: chromadb.ClientAPI, # type: ignore[reportPrivateImportUsage]
        collection_name: str,
    ) -> list[ChunkedText]:
    """Load all entries from a ChromaDB collection as a list of ChunkedTexts.
    (Primarily useful for testing and analysis.)"""
    collection = chroma_client.get_collection(name=collection_name)
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    assert data["embeddings"] is not None
    assert data["metadatas"] is not None
    assert data["documents"] is not None

    return [
        ChunkedText(text=doc, metadata=dict(meta), embedding=list(emb), id=_id)
        for doc, emb, meta, _id in zip(
            data['documents'],
            data['embeddings'],
            data['metadatas'],
            data['ids'],
        )
    ]


### Query-time: (r)etrieve, (a)ugment, (g)enerate

def build_context_injection(
    oai_client: OpenAI,
    collection: chromadb.Collection,
    user_query: str,
    n_results: int = config.N_RESULTS,
    d_threshold: float = config.DISTANCE_THRESHOLD,
) -> str:
    """Build context from vector neighbors to inject alongside a user query:
    Embed user query, retrieve `n_results` approximate-nearest chunks from
    ChromaDB, and format a context injection string for the system prompt."""
    q_embeds = embed_strings(oai_client, [user_query])
    q_results = collection.query(q_embeds, n_results=n_results)  # type: ignore inter-API mismatch

    # DISTANCE THRESHOLD FILTERING
    # TODO: consider implementing adaptive filtering for retrieval:
    #    keep the top result if closer than distance threshold, and
    #    keep subsequent results until distance > threshold, OR delta > max delta
    retrieved_chunks: list[dict] = []
    for id, meta, d, doc in zip(
        q_results['ids'][0],
        q_results['metadatas'][0],  # type: ignore (part of query()'s default include list)
        q_results['distances'][0],  # type: ignore (part of query()'s default include list)
        q_results['documents'][0],  # type: ignore (part of query()'s default include list)
        ):
        if d < d_threshold:
            status = 'Retrieved'
            retrieved_chunks.append({'id': id, 'metadata': meta, 'distance': d, 'document': doc})
        else:
            status = 'Discarded'
        logger.debug('%s %s[%s], d=%.6f > %s: %s',
                     status, meta.get('section'), meta.get('chunk'), d, d_threshold, doc)

    if not retrieved_chunks:
        return (
            "Retrieval results:\n"
            "No relevant biographical information was found for the following user query.\n"
            "Remember: speak naturally and don't reference the retrieval process.\n\n"
            "<retrieved_context></retrieved_context>"
        )

    tagged_chunks: list[str] = []
    for chunk in retrieved_chunks:
        tagged_chunks.append(
            f'  <chunk source="{chunk["id"]}">\n'
            f"    {chunk['document']}\n"
            f"  </chunk>"
        )

    return (
        "Retrieval results:\n"
        "The following biographical excerpts may be relevant the following user query.\n"
        "Use them, *if relevant*, to inform your response.\n"
        "Remember: speak naturally and don't reference the retrieval process.\n\n"
        f"<retrieved_context>\n{'\n'.join(tagged_chunks)}\n</retrieved_context>"
    )
