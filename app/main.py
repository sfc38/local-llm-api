import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ConnectError, RemoteProtocolError, TimeoutException

from app import services
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.database import (
    init_db, log_request, update_response, clear_requests,
    save_conversation, list_conversations, load_conversation,
    update_conversation, delete_conversation,
)
from app.ollama_client import OllamaClient
from app.schemas import (
    ChatRequest,
    ClassifyRequest,
    DescribeImageRequest,
    ExtractKeywordsRequest,
    GenerateRequest,
    SummarizeRequest,
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

init_db()

app = FastAPI(
    title="Local LLM API Playground",
    description="FastAPI wrapper for a locally running LLM via Ollama.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

_SKIP_LOG = {"/health", "/docs", "/openapi.json", "/redoc", "/models", "/reports",
             "/conversations", "/requests"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    body_bytes = await request.body()
    model_used = OLLAMA_MODEL
    prompt_preview = ""
    temperature = max_tokens = top_p = top_k = repeat_penalty = seed = num_ctx = None
    try:
        data = json.loads(body_bytes)
        model_used = data.get("model") or OLLAMA_MODEL
        prompt = data.get("prompt") or data.get("text") or ""
        messages = data.get("messages", [])
        if messages:
            last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            prompt = last_user or prompt
        prompt_preview = str(prompt)[:300]
        temperature = data.get("temperature")
        max_tokens = data.get("max_tokens")
        top_p = data.get("top_p")
        top_k = data.get("top_k")
        repeat_penalty = data.get("repeat_penalty")
        seed = data.get("seed")
        num_ctx = data.get("num_ctx")
    except Exception:
        pass

    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    logger.info("%s %s | %s | %dms", request.method, request.url.path, response.status_code, duration_ms)

    if request.url.path not in _SKIP_LOG and request.method == "POST":
        log_id = log_request(
            request.url.path, model_used, prompt_preview, duration_ms, response.status_code,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k,
            repeat_penalty=repeat_penalty, seed=seed, num_ctx=num_ctx,
        )  # duration_ms here = ttft_ms (time until stream starts)
        response.headers["X-Log-ID"] = str(log_id)

    return response


@app.exception_handler(ConnectError)
async def connect_error_handler(request, exc):
    return JSONResponse(status_code=503, content={"error": "Ollama is not reachable. Make sure Ollama is running."})


@app.exception_handler(TimeoutException)
async def timeout_error_handler(request, exc):
    return JSONResponse(status_code=504, content={"error": "Ollama timed out. The model may still be loading."})


@app.exception_handler(RemoteProtocolError)
async def remote_protocol_error_handler(request, exc):
    return JSONResponse(status_code=502, content={"error": "Ollama closed the connection unexpectedly."})


def _opts(r) -> dict:
    return {k: v for k, v in {
        "temperature": r.temperature,
        "max_tokens": r.max_tokens,
        "top_p": r.top_p,
        "top_k": r.top_k,
        "repeat_penalty": r.repeat_penalty,
        "seed": r.seed,
        "num_ctx": r.num_ctx,
    }.items() if v is not None}


@app.get("/health")
async def health():
    if not await ollama.is_reachable():
        raise HTTPException(status_code=503, detail="Ollama is not reachable.")
    return {"status": "ok", "ollama": "reachable", "model": OLLAMA_MODEL}


@app.get("/models")
async def get_models():
    try:
        return {"models": await ollama.get_models()}
    except (ConnectError, TimeoutException):
        raise HTTPException(status_code=503, detail="Ollama is not reachable.")


@app.get("/reports")
async def get_reports():
    from app.database import get_requests
    return {"requests": get_requests()}


@app.patch("/requests/{log_id}/response")
async def patch_response(log_id: int, body: dict):
    update_response(log_id, body.get("response_preview", ""),
                    body.get("response_time_ms"), body.get("ttft_ms"))
    return {"ok": True}


@app.delete("/requests")
async def delete_requests():
    clear_requests()
    return {"ok": True}


# ── Conversations ─────────────────────────────────────────────────────────────

@app.get("/conversations")
async def get_conversations():
    return {"conversations": list_conversations()}


@app.post("/conversations")
async def create_conversation(body: dict):
    conv_id = save_conversation(body.get("name", "Untitled"), body.get("messages", []))
    return {"id": conv_id}


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: int):
    conv = load_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.put("/conversations/{conv_id}")
async def put_conversation(conv_id: int, body: dict):
    update_conversation(conv_id, body.get("name", "Untitled"), body.get("messages", []))
    return {"ok": True}


@app.delete("/conversations/{conv_id}")
async def del_conversation(conv_id: int):
    delete_conversation(conv_id)
    return {"ok": True}


@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    return StreamingResponse(
        services.chat(ollama, messages, request.model, _opts(request)),
        media_type="text/event-stream",
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(
        services.generate(ollama, request.prompt, request.images, request.model, _opts(request)),
        media_type="text/event-stream",
    )


@app.post("/describe-image")
async def describe_image(request: DescribeImageRequest):
    return StreamingResponse(
        services.describe_image(ollama, request.image, request.prompt, request.model, _opts(request)),
        media_type="text/event-stream",
    )


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    return StreamingResponse(
        services.summarize(ollama, request.text, request.model, _opts(request)),
        media_type="text/event-stream",
    )


@app.post("/classify")
async def classify(request: ClassifyRequest):
    return StreamingResponse(
        services.classify(ollama, request.text, request.categories, request.model, _opts(request)),
        media_type="text/event-stream",
    )


@app.post("/extract-keywords")
async def extract_keywords(request: ExtractKeywordsRequest):
    return StreamingResponse(
        services.extract_keywords(ollama, request.text, request.model, _opts(request)),
        media_type="text/event-stream",
    )
