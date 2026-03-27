"""
Llama LLM web server with an OpenAI-compatible REST API.

Endpoints:
  GET  /v1/models                  - list available models
  POST /v1/completions             - text completion
  POST /v1/chat/completions        - chat completion (supports streaming)
  POST /v1/sql                     - natural language → DuckDB SQL (cached)
  POST /v1/embed                   - natural language query → embedding vector; pass hyde=false to avoid expanding via LLM first (HyDE)
  GET  /health                     - health check

Pass --model once per GGUF file to load multiple models simultaneously.
MODEL_PATH env var sets a single default model.
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
    model: str = Field(default="", description="Model to use (basename of the GGUF file, e.g. 'llama-3.1-8B.gguf'). Leave blank to use the first loaded model.")
    messages: list[Message]
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Maximum number of tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature — higher is more creative, lower is more deterministic.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling cutoff — only tokens comprising the top P probability mass are considered.")
    stream: bool = Field(default=False, description="Stream the response as server-sent events instead of returning it all at once.")
    stop: Optional[list[str]] = Field(default=None, description="Stop generation when any of these strings are produced.")


class CompletionRequest(BaseModel):
    model: str = Field(default="", description="Model to use (basename of the GGUF file, e.g. 'llama-3.1-8B.gguf'). Leave blank to use the first loaded model.")
    prompt: str = Field(description="The input text to continue.")
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Maximum number of tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature — higher is more creative, lower is more deterministic.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling cutoff — only tokens comprising the top P probability mass are considered.")
    stream: bool = Field(default=False, description="Stream the response as server-sent events instead of returning it all at once.")
    stop: Optional[list[str]] = Field(default=None, description="Stop generation when any of these strings are produced.")


class SqlRequest(BaseModel):
    model: str = Field(default="", description="Model to use (basename of the GGUF file, e.g. 'llama-3.1-8B.gguf'). Leave blank to use the first loaded model.")
    question: str = Field(description="Natural language question to convert into a DuckDB SQL query, e.g. 'find all TIFF files larger than 100 MB modified this week'.")
    schema: str = Field(default="", description="DuckDB CREATE TABLE statement(s) for your data. Providing this lets the model use the correct table and column names. Example: 'CREATE TABLE files (id INT, name VARCHAR, path VARCHAR, size BIGINT, modified TIMESTAMP);'")
    max_tokens: int = Field(default=200, ge=1, le=512, description="Maximum number of tokens to generate. SQL queries are short, so the default of 200 is usually plenty.")


class SearchRequest(BaseModel):
    query: str = Field(description="Natural language search query, e.g. 'fluorescence microscopy images of cell division'.")
    model: str = Field(default="", description="Model to use for HyDE expansion (basename of the GGUF file). Leave blank to use the first loaded model. Ignored when hyde=false.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of nearest neighbours the client should retrieve (informational — the server returns the vector and the client runs VSS locally in DuckDB-wasm).")
    hyde: bool = Field(default=True, description="When true, the LLM first writes a short hypothetical document that would answer the query, then that document is embedded instead of the raw query. This significantly improves recall when queries are short and indexed documents are long. Disable for raw query embedding.")


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

# Loaded models: maps model-id (file basename) → Llama instance
_models: dict[str, Llama] = {}
# Per-model inference lock — llama.cpp is single-threaded per instance,
# but different models can run concurrently.
_model_locks: dict[str, asyncio.Lock] = {}


def get_llm(name: str | None = None) -> tuple[str, Llama, asyncio.Lock]:
    """Return (resolved_name, llm, lock) for the requested model.

    Pass None or "" to use the first loaded model (default).
    Raises 404 if a non-empty name is given that doesn't match any loaded model.
    """
    if not _models:
        raise HTTPException(status_code=503, detail="No model loaded")
    if not name:
        key = next(iter(_models))
        return key, _models[key], _model_locks[key]
    if name not in _models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Loaded models: {list(_models)}",
        )
    return name, _models[name], _model_locks[name]


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
        "models": {
            name: {"queue_locked": _model_locks[name].locked()}
            for name in _models
        },
    }


@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": now, "owned_by": "local"}
            for name in _models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    _, model, inference_lock = get_llm(req.model or None)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    log.info("chat/completions → %d messages, max_tokens=%d", len(messages), req.max_tokens)

    t0 = time.perf_counter()
    async with inference_lock:
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
    _, model, inference_lock = get_llm(req.model or None)

    log.info("completions → prompt_len=%d chars, max_tokens=%d", len(req.prompt), req.max_tokens)
    t0 = time.perf_counter()
    async with inference_lock:
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
    resolved_name, model, inference_lock = get_llm(req.model or None)
    cache_key = (resolved_name, req.schema, req.question.strip().lower())

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
    async with inference_lock:
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

    With hyde=true the LLM first generates a short hypothetical document that
    would answer the query; that document is embedded instead of the raw query.
    This improves recall when queries are much shorter than indexed documents.

    Returns: { "embedding": [0.123, ...], "dim": 384, "hyde_doc": "..." | null }
    """
    text_to_embed = req.query
    hyde_doc: Optional[str] = None

    if req.hyde:
        _, model, inference_lock = get_llm(req.model or None)
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short, factual document (2-4 sentences) that directly answers "
                    "the user's query. Output only the document text, no commentary."
                ),
            },
            {"role": "user", "content": req.query},
        ]
        log.info("hyde → generating hypothetical doc for: %s", req.query[:80])
        t_hyde = time.perf_counter()
        async with inference_lock:
            hyde_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.create_chat_completion(
                    messages=messages,
                    max_tokens=120,
                    temperature=0.3,
                    top_p=0.95,
                ),
            )
        hyde_doc = hyde_response["choices"][0]["message"]["content"].strip()
        log.info("hyde → doc in %.2fs: %s", time.perf_counter() - t_hyde, hyde_doc[:80])
        text_to_embed = hyde_doc

    async with _embed_lock:
        embed_model = await asyncio.get_event_loop().run_in_executor(
            None, get_embed_model
        )
        t0 = time.perf_counter()
        vec = await asyncio.get_event_loop().run_in_executor(
            None, lambda: embed_model.encode(text_to_embed).tolist()
        )
    elapsed = time.perf_counter() - t0
    log.info("embed '%s' → dim=%d in %.3fs", req.query[:60], len(vec), elapsed)
    return {"embedding": vec, "dim": len(vec), "hyde_doc": hyde_doc}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama LLM web server")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="PATH",
        default=None,
        help="Path to a GGUF model file (repeat to load multiple models)",
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

    env_path = os.environ.get("MODEL_PATH", "")
    model_paths: list[str] = args.models or ([env_path] if env_path else [])
    if not model_paths:
        raise SystemExit(
            "Error: no model specified. Use --model <path-to-model.gguf> "
            "(repeat for multiple models) or set the MODEL_PATH env var."
        )

    for path in model_paths:
        name = os.path.basename(path)
        print(f"Loading model: {path}")
        _models[name] = Llama(
            model_path=path,
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
        _model_locks[name] = asyncio.Lock()
        print(f"  ✓ {name}")

    print(f"Server running on http://{args.host}:{args.port}")

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
