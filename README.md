<p align="center">
  <img src="assets/logo.png" width="320" />
</p>

# Jeremy's Digital Twin

A RAG-powered chatbot that responds as a digital version of Jeremy Dolan. Built with Gradio, OpenAI, and ChromaDB.

## How it works

Biographical facts in `data/biography.txt` are chunked, embedded (OpenAI `text-embedding-3-large`), and stored in a (graph-based) vector index (ChromaDB). At runtime, user messages are embedded into the same vector space and approximate nearest-neighbor search identifies potentially relevant chunks. These chunks are injected as context alongside a system prompt that instructs the LLM (OpenAI `gpt-5.2`/Responses API) to respond in Jeremy's voice (through a Gradio `ChatInterface`).

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
scripts/         — Utility scripts (e.g., rebuild vectors)
```

## Vector store

To build the vector store (after editing `data/biography.txt`):

```bash
python scripts/build_vectors.py
```

## Running locally

```bash
python3.13 -m venv .venv
# (as of chromadb 1.5.2, python3.14 is not supported)
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# ...fill in API keys
LOG_LEVEL=DEBUG python app.py
# ...or `LOG_LEVEL=DEBUG gradio app.py` to use in Gradio's 'watch mode'
```

## Deploying to Hugging Face Spaces

1) Create a (private) HF data repo for the chromadb database, and update config.HUGGINGFACE_DATASET_REPO.
2) Add `HF_TOKEN`, `OPENAI_API_KEY`, `PUSHOVER_USER`, and `PUSHOVER_TOKEN` as secrets in the HF Space.
3) Run scripts/hf_deploy.sh, which isn't written yet, so in the meantime, paste the following and hope for the best.

```sh
git diff --staged --quiet &&
mv README.md README._ &&
mv README.hf-spaces.yaml.md README.md &&
git add README.md &&
git rm --cached assets/demo.png &&
git push hf $(git commit-tree $(git write-tree) -m 'deploy'):refs/heads/main --force &&
git restore --staged README.md assets/demo.png &&
mv README.md README.hf-spaces.yaml.md &&
mv README._ README.md
```
Note: This does two things: Swaps the normal README with a yaml README to configure the Space (because the yaml displays catastrophically on Github); and removes all history of demo.png (because Spaces won't accept "large" binary files, so we push an orphan commit that has no reference to demo.png).

The first line ensures we only start if no changes are staged.
