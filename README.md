<p align="center">
  <img src="assets/logo.png" width="320" />
</p>

# Jeremy's Digital Twin

A RAG-powered chatbot that responds as a digital version of Jeremy Dolan. Built with Gradio, OpenAI, and ChromaDB.  
Give it a spin: https://virtual.jeremydolan.net/

## How it works

I gave Claude (Opus 4.6) my resume and a brief personal summary, and prompted it to conduct a structured interview that would surface and fill gaps in that initial information. This yielded a purpose-built source document optimized for chunking and retrieval ([example](data/biography.example.txt)), with optional per-statement instructions on how and when I want that information presented. (Effectively, a chatbot helped turn me into a chatbot.)

This biography was then chunked, embedded (OpenAI `text-embedding-3-large`), and stored in a vector index (ChromaDB). At runtime, the user's message is embedded, and vector similarity search retrieves chunks within a fixed distance. These chunks (with their metadata) are included as context alongside the user query. The [system prompt](prompts.py) instructs the LLM (OpenAI `gpt-5.2`) to respond in my voice, using the biography chunks when relevant. Conversation state and UI are managed through a lightly customized Gradio `ChatInterface`.

The system notifies me in real time of any urgent or interesting developments during conversations (Pushover API).
<!-- TODO: schedule a meeting via Calendly API? -->

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
tests/           — pytests, courtesy of Claude
```

## Running locally

```sh
python3.13 -m venv .venv         # as of chromadb 1.5.2, python3.14 is not supported
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # ...and add API keys
# Create data/biography.txt      # see biography.example.txt for a guide
python scripts/build-vectors.py  # rebuild the vector index from biography.txt
# Personalize prompts.py         # choose style preferences, add a baseline biography
LOG_LEVEL=DEBUG python app.py    # ...or `gradio app.py` to use Gradio's 'watch mode'
```

## Deploying to Hugging Face Spaces

1) Create a HF Space; add it to `git` as an additional remote:  
   `git remote add hf https://huggingface.co/spaces/[user]/[space]`
2) Add `HF_TOKEN`, `OPENAI_API_KEY`, `PUSHOVER_USER`, and `PUSHOVER_TOKEN` as secrets in the Space.
3) Create a (private) HF data repo to store the chromadb index. Upload the index. Update config.HUGGINGFACE_DATASET_REPO.
4) Run `scripts/hf-deploy` to push an orphan commit to the 'hf' remote:
    * Removes assets/demo.png from the tree (because HF Spaces rejects large binaries)
    * Applies deploy/*.patch (Adds YAML frontmatter to README.md, which otherwise displays catastrophically on Github; also some security-through-obscurity additions to the API endpoint which I can't publish, but gradio-app/gradio#13051 describes the problem.)
