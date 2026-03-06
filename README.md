# Jeremy's Digital Twin

A RAG-powered chatbot that responds as a digital version of Jeremy Dolan. Built with Gradio, OpenAI, and ChromaDB.

## How it works

Biographical facts in `data/biography.txt` are chunked, embedded (OpenAI `text-embedding-3-large`), and stored in a vector database (ChromaDB). At runtime, user messages are embedded into the same vector space and approximate nearest-neighbor search identifies potentially relevant chunks. These chunks are injected as context alongside a system prompt that instructs the LLM to respond in Jeremy's voice (through a Gradio `ChatInterface`).

The LLM can also use tool calling to schedule a meeting with Jeremy (Calendly API), or send him a push notification (Pushover API).

## Data generation

Claude (Opus 4.6) was given my resume and a brief personal summary, then prompted to conduct a structured interview that would iteratively surface and fill gaps in that initial information. This yielded a purpose-built source document optimized for chunking and retrieval. (Effectively, a chat bot helped turn me into a chat bot.)

## Project structure

```
app.py           — Hugging Face Spaces/Gradio entry point
config.py        — Configuration (models, thresholds, paths)
inference.py     — LLM response loop with tool call processing
rag.py           — Chunking, embedding, retrieval, context injection
tools.py         — Tool registry and implementations
prompts.py       — System message
data/            — biography.txt source data
chromadb/        — Vector store
scripts/         — Utility scripts (e.g., rebuild vectors)
```

## Vector store

To rebuild the vector store (after editing `data/biography.txt`):

```bash
python scripts/build_vectors.py
```

## Running locally

```bash
python3.13 -m venv .venv
# chromadb does not support 3.14
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill keys in .env
python app.py
```

## Deploying to Hugging Face Spaces

Add `OPENAI_API_KEY`, `PUSHOVER_USER`, and `PUSHOVER_TOKEN` as Space secrets in Settings.
[FIXME]
