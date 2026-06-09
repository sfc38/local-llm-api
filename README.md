# Local LLM API Playground

A FastAPI project that exposes a local LLM through API endpoints using Ollama and Qwen2.5-VL. Demonstrates Python API creation, JSON request/response design, streaming responses, and local LLM inference without paid API calls.

---

## Goal

Learn API design and local LLM integration by building a clean Python backend that accepts prompts and images, and returns AI-generated responses — entirely offline, no paid services required.

---

## Features

**API endpoints**
- `GET /health` — check that the API and Ollama are reachable
- `POST /chat` — multi-turn conversation with full history
- `POST /generate` — single prompt with optional images
- `POST /describe-image` — send a base64 image and get a description
- `POST /summarize` — summarize a block of text
- `POST /classify` — classify text into a provided set of categories
- `POST /extract-keywords` — extract key terms from text
- Streaming responses via Server-Sent Events (SSE)
- All 7 sampling hyperparameters supported per request

**Streamlit UI**
- Chat with persistent conversation history and *Thinking…* indicator
- Save / load / delete named conversations (stored in SQLite — works like ChatGPT history)
- Attach images or PDFs to chat messages via a 📎 popover next to the input
- Text PDFs: all pages extracted and sent as context
- Scanned PDFs: pages rendered as JPEG images and sent to the vision model
- Reports page: prompts, responses, hyperparameters, TTFT + total response time per request, endpoint breakdown charts, CSV export
- Model selector dropdown (live from Ollama), 7 hyperparameter sliders with explanations

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| API framework | FastAPI | MIT |
| Server | Uvicorn | BSD |
| LLM runner | Ollama | MIT |
| Default model | Qwen2.5-VL 3B | Apache 2.0 |
| HTTP client | httpx | BSD |
| Validation | Pydantic | MIT |
| Frontend | Streamlit | Apache 2.0 |
| PDF text extraction | pypdf | BSD-3-Clause |
| Scanned PDF rendering | pypdfium2 | Apache 2.0 |
| Language | Python 3.10+ | PSF |

All tools are free and open-source with licenses compatible with commercial use.

---

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed — download from [ollama.com](https://ollama.com) (use the official Mac/Windows app, not Homebrew)
3. **Qwen2.5-VL 3B model** pulled:

```bash
ollama pull qwen2.5vl:3b
```

---

## Setup

```bash
git clone https://github.com/sfc38/local-llm-api.git
cd local-llm-api

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

**Step 1 — Ollama is already running** if you see the llama icon in your menu bar. If not, open the Ollama app.

**Step 2 — Start the API:**

```bash
uvicorn app.main:app --reload
```

**Step 3 — Choose your interface:**

- **Swagger UI** (API testing): `http://127.0.0.1:8000/docs`
- **Streamlit UI** (browser playground): `streamlit run streamlit_app.py`

To access from another device on the same Wi-Fi:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

---

## Docker

Build and run the API in a container (Ollama must be running on the host):

```bash
docker build -t local-llm-api .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 local-llm-api
```

---

## Example Requests

**Multi-turn chat:**

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Nice to meet you, Alice!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

The model sees the full conversation history each time — that is how memory works.

---

**Generate with temperature control:**

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what RAG means.", "temperature": 0.5, "max_tokens": 200}'
```

**Describe an image (Python):**

```python
import base64, requests

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

requests.post("http://localhost:8000/describe-image", json={
    "image": b64,
    "prompt": "What is in this image?"
})
```

---

## Request Parameters

All endpoints accept these optional fields:

| Field | Type | Range | Description |
|---|---|---|---|
| `model` | string | — | Override the default model for this request |
| `temperature` | float | 0–2 | Randomness. Lower = more focused, higher = more creative |
| `max_tokens` | int | 1–8192 | Maximum tokens to generate |
| `top_p` | float | 0–1 | Nucleus sampling threshold |
| `top_k` | int | 1–100 | Candidate pool size per token |
| `repeat_penalty` | float | 0.5–2 | Penalise recently used tokens |
| `seed` | int | — | Fixed seed for reproducible outputs |
| `num_ctx` | int | 512–32768 | Context window size in tokens |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | Default model |

For Oracle Cloud deployment: set `OLLAMA_BASE_URL=http://<server-ip>:11434`.

---

## Project Structure

```
local-llm-api/
  app/
    main.py          — FastAPI app, routes, middleware, error handlers, CORS
    config.py        — Environment variable config
    schemas.py       — Pydantic request models (LLMBase + 7 hyperparameters)
    ollama_client.py — Reusable async HTTP client for Ollama
    services.py      — Prompt templates and SSE streaming
    database.py      — SQLite helpers (requests log + conversations)
  tests/
    test_api.py      — Integration tests (8 tests, all passing)
  streamlit_app.py   — Browser-based UI playground
  Dockerfile         — Container for the FastAPI server
  requirements.txt
  logs/
    requests.db      — SQLite: request log + saved conversations (gitignored)
    api.log          — Text log (gitignored)
```

---

## Running Tests

Requires both Ollama and the FastAPI server to be running.

```bash
pytest tests/ -v
```

---

## Future Improvements

- Oracle Cloud deployment (Ollama + FastAPI on Always Free Ampere ARM instance)
- PDF chunking for documents longer than 3 pages (scanned) or very long text PDFs
- Conversation history truncation at context window limit
- Rate limiting and API key authentication

---

## License

MIT — free to use, modify, and distribute.
