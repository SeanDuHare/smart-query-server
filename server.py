"""
Llama LLM web server with an OpenAI-compatible REST API.

Endpoints:
  GET  /v1/models                  - list available models
  POST /v1/completions             - text completion
  POST /v1/chat/completions        - chat completion (supports streaming)
  POST /v1/sql                     - natural language → DuckDB SQL (cached)
  POST /v1/embed                   - natural language query → embedding vector (client runs VSS in DuckDB-wasm)
  GET  /health                     - health check

Set the model path with the --model flag or MODEL_PATH env var.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Iterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

log = logging.getLogger("llm-server")

# Requests slower than this threshold get a WARNING log
SLOW_REQUEST_THRESHOLD_S = 10.0

# Embedding model for VSS — loaded lazily on first /v1/embed call
# Uses a small 384-dim model (~90MB) — fast on CPU
_embed_model: Optional[SentenceTransformer] = None
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_lock = asyncio.Lock()


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Embedding model loaded")
    return _embed_model

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama"
    messages: list[Message]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class CompletionRequest(BaseModel):
    model: str = "llama"
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class SqlRequest(BaseModel):
    question: str
    # Paste your DuckDB CREATE TABLE statements here so the model generates
    # valid column/table names. Example:
    #   schema: "CREATE TABLE files (id INT, name VARCHAR, path VARCHAR, size BIGINT, modified TIMESTAMP);"
    schema: str = ""
    max_tokens: int = Field(default=200, ge=1, le=512)


class SearchRequest(BaseModel):
    # Natural language query to embed — the client runs VSS in DuckDB-wasm
    query: str
    # Return top_k nearest neighbors if the client also sends candidate vectors
    # (optional — if omitted the server just returns the embedding vector)
    top_k: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Llama LLM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    method = request.method
    status = response.status_code
    msg = f"{method} {path} → {status} ({elapsed:.2f}s)"
    if elapsed >= SLOW_REQUEST_THRESHOLD_S:
        log.warning("SLOW %s", msg)
    else:
        log.info(msg)
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response

llm: Optional[Llama] = None
model_path: str = ""

# Serializes inference calls so concurrent requests queue instead of
# crashing — llama.cpp is single-threaded per model instance.
_inference_lock = asyncio.Lock()


def get_llm() -> Llama:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_chat(response_iter: Iterator) -> Iterator[str]:
    for chunk in response_iter:
        delta = chunk["choices"][0]["delta"]
        content = delta.get("content", "")
        data = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "created": chunk["created"],
            "model": chunk["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_completion(response_iter: Iterator) -> Iterator[str]:
    for chunk in response_iter:
        text = chunk["choices"][0].get("text", "")
        data = {
            "id": chunk["id"],
            "object": "text_completion",
            "created": chunk["created"],
            "model": chunk["model"],
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": chunk["choices"][0].get("finish_reason"),
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": llm is not None,
        "queue_locked": _inference_lock.locked(),
    }


@app.get("/v1/models")
def list_models():
    name = os.path.basename(model_path) if model_path else "llama"
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = get_llm()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    log.info("chat/completions → %d messages, max_tokens=%d", len(messages), req.max_tokens)

    t0 = time.perf_counter()
    async with _inference_lock:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop or [],
                stream=req.stream,
            ),
        )
    elapsed = time.perf_counter() - t0
    log.warning("SLOW inference %.2fs", elapsed) if elapsed >= SLOW_REQUEST_THRESHOLD_S else log.info("inference done in %.2fs", elapsed)

    if req.stream:
        return StreamingResponse(
            _stream_chat(response),
            media_type="text/event-stream",
        )

    return response


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    model = get_llm()

    log.info("completions → prompt_len=%d chars, max_tokens=%d", len(req.prompt), req.max_tokens)
    t0 = time.perf_counter()
    async with _inference_lock:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.create_completion(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop or [],
                stream=req.stream,
            ),
        )
    elapsed = time.perf_counter() - t0
    log.warning("SLOW inference %.2fs", elapsed) if elapsed >= SLOW_REQUEST_THRESHOLD_S else log.info("inference done in %.2fs", elapsed)

    if req.stream:
        return StreamingResponse(
            _stream_completion(response),
            media_type="text/event-stream",
        )

    return response


# ---------------------------------------------------------------------------
# DuckDB text-to-SQL endpoint
# ---------------------------------------------------------------------------

# Simple LRU-style cache: maps (schema, question) → sql string
_sql_cache: dict[tuple[str, str], str] = {}
SQL_CACHE_MAX = 256


@app.post("/v1/sql")
async def text_to_sql(req: SqlRequest):
    """
    Convert a natural language question into a DuckDB SQL query.
    Optimized for speed: low token budget, temperature=0, cached results.

    Returns: { "sql": "SELECT ..." }
    """
    model = get_llm()
    cache_key = (req.schema, req.question.strip().lower())

    if cache_key in _sql_cache:
        log.info("sql cache hit for: %s", req.question[:60])
        return {"sql": _sql_cache[cache_key], "cached": True}

    schema_block = f"\nDatabase schema:\n{req.schema}\n" if req.schema else ""
    system_prompt = (
        "You are a DuckDB SQL expert. "
        "Given a question, output ONLY a single valid DuckDB SQL query with no explanation, "
        "no markdown, no backticks. Output the raw SQL only."
    )
    messages = [
        {"role": "system", "content": system_prompt + schema_block},
        {"role": "user", "content": req.question},
    ]

    log.info("sql → question: %s", req.question[:80])
    t0 = time.perf_counter()
    async with _inference_lock:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=0.0,   # deterministic — SQL should be exact
                top_p=1.0,
                stop=[";"],        # stop after first statement
            ),
        )
    elapsed = time.perf_counter() - t0
    log.warning("SLOW sql inference %.2fs", elapsed) if elapsed >= SLOW_REQUEST_THRESHOLD_S else log.info("sql inference done in %.2fs", elapsed)

    sql = response["choices"][0]["message"]["content"].strip()

    # Evict oldest entry if cache is full
    if len(_sql_cache) >= SQL_CACHE_MAX:
        _sql_cache.pop(next(iter(_sql_cache)))
    _sql_cache[cache_key] = sql

    return {"sql": sql, "cached": False}


# ---------------------------------------------------------------------------
# Embedding endpoint — client runs VSS in DuckDB-wasm
# ---------------------------------------------------------------------------

@app.post("/v1/embed")
async def embed_query(req: SearchRequest):
    """
    Embed a natural language query and return the float vector.

    The client (DuckDB-wasm in the browser) uses this vector to run a
    VSS similarity search locally:

        SELECT *, array_distance(embedding, ?::FLOAT[384]) AS _distance
        FROM files ORDER BY _distance LIMIT 10;

    Returns: { "embedding": [0.123, ...], "dim": 384 }
    """
    async with _embed_lock:
        embed_model = await asyncio.get_event_loop().run_in_executor(
            None, get_embed_model
        )
        t0 = time.perf_counter()
        vec = await asyncio.get_event_loop().run_in_executor(
            None, lambda: embed_model.encode(req.query).tolist()
        )
    elapsed = time.perf_counter() - t0
    log.info("embed '%s' → dim=%d in %.3fs", req.query[:60], len(vec), elapsed)
    return {"embedding": vec, "dim": len(vec)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama LLM web server")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", ""),
        help="Path to the GGUF model file (or set MODEL_PATH env var)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--n-ctx", type=int, default=2048,
        help="Context window per request (default: 2048 — keep small for search tasks)",
    )
    parser.add_argument(
        "--n-batch", type=int, default=512,
        help="Prompt processing batch size (default: 512)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (default: 0 = CPU only)",
    )
    parser.add_argument(
        "--threads", type=int, default=os.cpu_count(), help="CPU threads (default: all cores)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.model:
        raise SystemExit(
            "Error: no model specified. Use --model <path-to-model.gguf> "
            "or set the MODEL_PATH environment variable."
        )

    model_path = args.model
    print(f"Loading model: {model_path}")

    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.threads,
        # Keep KV cache in fp16 to halve its memory usage
        f16_kv=True,
        # Disable memory-mapping if the model fits in RAM — faster random access
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

    print(f"Model loaded. Server running on http://{args.host}:{args.port}")

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)-8s %(name)s  %(message)s", "datefmt": "%H:%M:%S"},
        },
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"},
        },
        "loggers": {
            "llm-server": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_config=log_config)
