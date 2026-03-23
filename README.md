<p align="center">
  <img src="assets/logo.png" width="320" />
</p>

# Jeremy's Digital Twin

A RAG-powered chatbot that responds as a digital version of Jeremy Dolan. Built with Gradio, OpenAI, and ChromaDB.

## How it works

Biographical facts in `data/biography.txt` are chunked, embedded (OpenAI `text-embedding-3-large`), and stored in a (graph-based) vector index (ChromaDB). At runtime, user messages are embedded into the same vector space and approximate nearest-neighbor search identifies potentially relevant chunks. These chunks are injected as context alongside a system prompt that instructs the LLM (OpenAI `gpt-5.2`) to respond in Jeremy's voice (through a Gradio `ChatInterface`).

The LLM can also use tool calling to schedule a meeting with Jeremy (Calendly API), or send him a push notification (Pushover API).

## Data generation

Claude (Opus 4.6) was given my resume and a brief personal summary, then prompted to conduct a structured interview that would iteratively surface and fill gaps in that initial information. This yielded a purpose-built source document optimized for chunking and retrieval. (Effectively, a chat bot helped turn me into a chat bot.)

<p align="center">
  <img src="assets/demo.png" width="755" />
</p>

## Project structure

```
app.py           — Gradio/Hugging Face Spaces entry point
config.py        — Configuration (models, thresholds, paths)
inference.py     — LLM response loop with tool call processing
rag.py           — Chunking, embedding, retrieval, context injection
tools.py         — Tool registry and implementations
prompts.py       — System message
assets/          — Logo, avatar, and favicon images
data/            — biography.txt source data
chromadb/        — Vector store
scripts/         — Utility scripts (e.g., rebuild vectors, deploy)
deploy/          — Patches applied at deploy time for HF Spaces
```

## Vector store

To build the vector store (after editing `data/biography.txt`):

```sh
python scripts/build-vectors.py
```

## Running locally

```sh
python3.13 -m venv .venv         # as of chromadb 1.5.2, python3.14 is not supported
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # ...and add API keys
LOG_LEVEL=DEBUG python app.py    # ...or `gradio app.py` to use Gradio's 'watch mode'
```

## Deploying to Hugging Face Spaces

1) Create a HF Space; add it to `git` as an additional remote:  
   `git remote add hf https://huggingface.co/spaces/[user]/[space]`
2) Add `HF_TOKEN`, `OPENAI_API_KEY`, `PUSHOVER_USER`, and `PUSHOVER_TOKEN` as secrets in the Space.
3) Create a (private) HF data repo for the chromadb database; update config.HUGGINGFACE_DATASET_REPO.
4) Run `scripts/hf-deploy` to push an orphan commit to the 'hf' remote:
    * Removes assets/demo.png from the tree (because HF Spaces rejects large binaries)
    * Applies deploy/*.patch (Adds YAML frontmatter to README.md, which otherwise displays catastrophically on Github; also some security-through-obscurity additions to the API endpoint which I can't publish, but gradio-app/gradio#13051 describes the problem.)
