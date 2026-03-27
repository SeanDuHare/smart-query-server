# Llama LLM Web Server

A lightweight OpenAI-compatible REST API server for running Llama (GGUF) models locally.

## Requirements

- Python 3.10+
- A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))

## Setup

```bash
cd test-llm-server

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU)
pip install -r requirements.txt

# linux/ubuntu (required packages)
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    python3-dev \
    python3-venv

## required for ubuntu 18.04
# Ubuntu 18.04 ships with GCC 7, which has incomplete C++17 support. Recent llama-cpp-python requires GCC 9+

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-9 g++-9

# if that fails try this: Disable the failing Chrome repo temporarily
sudo mv /etc/apt/sources.list.d/google-chrome.list /etc/apt/sources.list.d/google-chrome.list.bak

sudo apt update
sudo apt install -y gcc-9 g++-9

# Restore it after
sudo mv /etc/apt/sources.list.d/google-chrome.list.bak /etc/apt/sources.list.d/google-chrome.list

# Test that it worked via:
g++-9 --version # should show 9.x

# If sqllite3 is not found run this:
#The system Python was compiled before libsqlite3-dev was installed, so it can't pick it up retroactively. You need a Python that was compiled with SQLite support.
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

deactivate
rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Then reinstall with the newer compiler
CC=gcc-9 CXX=g++-9 pip install llama-cpp-python

# Thennnnnnnnn
sudo apt install -y libsqlite3-dev

# macOS Metal GPU acceleration (optional)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python

# NVIDIA CUDA GPU acceleration (optional)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --force-reinstall llama-cpp-python

# NVIDIA GPU on Ubuntu (optional)
# TODO:?????
sudo apt install -y nvidia-cuda-toolkit
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --force-reinstall llama-cpp-python

```

## Download a model

Any GGUF model works. Example using a small Llama 3 instruct model:

```bash
# Using huggingface-hub
pip install huggingface-hub
# huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
#     Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
#     pip install huggingface-hub
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## Run the server

```bash
# python server.py --model ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
python server.py --model ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `$MODEL_PATH` | Path to the GGUF model file |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | TCP port |
| `--n-ctx` | `2048` | Context window size in tokens (keep small for search tasks) |
| `--n-batch` | `512` | Prompt processing batch size |
| `--n-gpu-layers` | `0` | Layers to offload to GPU (0 = CPU only) |
| `--threads` | all cores | CPU inference threads |

You can also set `MODEL_PATH` as an environment variable instead of using `--model`.

## API

The server exposes an **OpenAI-compatible** API plus two specialized endpoints for DuckDB file search.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check + queue status |
| `/v1/models` | GET | List loaded model |
| `/v1/chat/completions` | POST | Chat (OpenAI-compatible, streaming supported) |
| `/v1/completions` | POST | Raw text completion |
| `/v1/sql` | POST | Natural language → DuckDB SQL (cached) |
| `/v1/embed` | POST | Natural language query → embedding vector (client runs VSS in DuckDB-wasm) |

---

### Health check

```
GET /health
```

Response includes `queue_locked` — if `true`, an inference is in progress and new requests are queued.

### Chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'
```

### Text-to-SQL (`/v1/sql`)

Converts a natural language question into a valid DuckDB SQL query.
Results are cached — identical questions return instantly on subsequent calls.

```bash
curl http://localhost:8000/v1/sql \
  -H "Content-Type: application/json" \
  -d '{
    "question": "find all tiff files larger than 100MB modified in the last 7 days",
    "schema": "CREATE TABLE files (id INT, name VARCHAR, path VARCHAR, size BIGINT, modified TIMESTAMP);"
  }'
```

Response:
```json
{
  "sql": "SELECT * FROM files WHERE name LIKE '%.tiff' AND size > 104857600 AND modified >= current_date - INTERVAL 7 DAYS",
  "cached": false
}
```

Your client then runs that SQL directly against DuckDB — the model only generates ~30 tokens instead of
a full prose answer, making it 5–10x faster than open-ended chat.

| Field | Type | Description |
|---|---|---|
| `question` | string | Natural language question |
| `schema` | string | DuckDB `CREATE TABLE` statement(s) for accurate column names |
| `max_tokens` | int | Max tokens to generate (default: 200) |

### Embed query for VSS (`/v1/embed`)

Since DuckDB runs in the browser via **DuckDB-wasm**, the server never touches your data.
Instead, the server embeds the natural language query into a float vector and returns it —
the client runs the actual VSS query locally in DuckDB-wasm.

```bash
curl http://localhost:8000/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"query": "microscopy images of cell division"}'
```

Response:
```json
{
  "embedding": [0.0231, -0.0842, 0.1103, "..."],
  "dim": 384
}
```

The client then runs this in DuckDB-wasm:
```sql
SELECT *, array_distance(embedding, ?::FLOAT[384]) AS _distance
FROM files
ORDER BY _distance
LIMIT 10;
```
passing the `embedding` array as the bound parameter.

| Field | Type | Description |
|---|---|---|
| `query` | string | Natural language search query |

#### Setting up embeddings in DuckDB-wasm

Pre-populate the embedding column at index time using
[Transformers.js](https://huggingface.co/docs/transformers.js) directly in the browser
(same model = compatible vectors, no server round-trip at write time):

```js
import { pipeline } from "@xenova/transformers";

const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

const output = await embedder("mitosis z-stack tiff", { pooling: "mean", normalize: true });
const vec = Array.from(output.data); // Float32Array → plain array

// INSERT INTO files (..., embedding) VALUES (..., vec)
await db.query(`INSERT INTO files (name, path, embedding) VALUES (?, ?, ?)`, [name, path, vec]);
```

> **Important:** use the same model on both sides (`all-MiniLM-L6-v2` / `Xenova/all-MiniLM-L6-v2`)
> so server-generated query vectors and client-stored file vectors are in the same embedding space.

---

## Client examples

### Python (openai library)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Text-to-SQL + DuckDB-wasm (JavaScript)

```js
// 1. Ask the server to generate SQL
const { sql } = await fetch("http://localhost:8000/v1/sql", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "show me the 10 largest tiff files modified this week",
    schema: "CREATE TABLE files (id INT, name VARCHAR, path VARCHAR, size BIGINT, modified TIMESTAMP);",
  }),
}).then(r => r.json());

// 2. Run it locally in DuckDB-wasm (no server involved)
const results = await db.query(sql);
console.log(results.toArray());
```

### Semantic search + DuckDB-wasm (JavaScript)

```js
// 1. Get the embedding vector from the server
const { embedding } = await fetch("http://localhost:8000/v1/embed", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "fluorescence microscopy z-stack" }),
}).then(r => r.json());

// 2. Run VSS locally in DuckDB-wasm
const stmt = await db.prepare(
  `SELECT name, path, array_distance(embedding, ?::FLOAT[384]) AS _distance
   FROM files ORDER BY _distance LIMIT 10`
);
const results = await stmt.query(embedding);
console.log(results.toArray());
```
