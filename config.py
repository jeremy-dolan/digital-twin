from pathlib import Path

# for typing
from openai.types.shared_params import Reasoning
from chromadb.config import Settings
from chromadb.api.collection_configuration import CreateCollectionConfiguration

### paths
BASE_DIR = Path(__file__).resolve().parent
BIOGRAPHY_TXT = BASE_DIR / 'data' / 'biography.txt'
CHROMA_PATH = BASE_DIR / 'chromadb'

### OpenAI -- API_KEY in .env
# INFERENCE_MODEL='gpt-5-mini'              # 2.5 million tokens/day for usage sharing
INFERENCE_MODEL = 'gpt-5.2'
#EMBEDDING_MODEL = 'text-embedding-3-small' # 1536 dimensions, max 8192 tokens
EMBEDDING_MODEL = 'text-embedding-3-large'  # 3072 dimensions, max 8192 tokens
REASONING = Reasoning(summary='auto')       # XXX change to 'none' for production

### ChromaDB
CHROMA_COLLECTION_NAME = 'bio_facts_large'
CHROMA_COLLECTION_CONFIG = CreateCollectionConfiguration(hnsw={"space": "cosine"})
CHROMA_CLIENT_SETTINGS = Settings(anonymized_telemetry=False) # don't send usage

### RAG retrieval tuning
N_RESULTS = 10
DISTANCE_THRESHOLD = 0.825 # For space=cosine, Chroma uses distance = (1 - cosine_similarity)
                           # Range: 0 (identical) to 2 (opposite); 1 is orthogonal.
                           # OpenAI's embedding models norm vectors to length 1, so equally:
                           #    cosine distance = (1 - dot product)
                           #    cosine distance = (1/2 * squared Euclidean distance)
                           # TODO: sample more queries, label chunks, compare distribution

### tool processing
MAX_SEQUENTIAL_TOOL_CALLS = 4 # generous; prevent runaway tool recursion

### 'Pushover' service (for send_notification tool) -- API USER/TOKEN is in .env
PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"
