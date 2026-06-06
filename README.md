# Local LLM API Playground

A FastAPI project that exposes a local LLM through API endpoints using Ollama and Qwen2.5. The project demonstrates Python API creation, JSON request/response design, and local LLM inference without paid API calls.

---

## Goal

Learn API design and local LLM integration by building a clean Python backend that accepts prompts and returns AI-generated responses — entirely offline, no paid services required.

---

## Features

- `GET /health` — check that the API and Ollama are reachable
- `POST /generate` — send a prompt, get a response from a local LLM
- `POST /summarize` — summarize a block of text
- `POST /classify` — classify text into a provided set of categories
- `POST /extract-keywords` — extract key terms from text
- Interactive API docs at `http://127.0.0.1:8000/docs` (Swagger UI, built-in)

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| API framework | FastAPI | MIT |
| Server | Uvicorn | BSD |
| LLM runner | Ollama | MIT |
| Default model | Qwen2.5 3B | Apache 2.0 |
| Validation | Pydantic | MIT |
| Language | Python 3.10+ | PSF |

All tools are free and open-source with licenses compatible with commercial use.

---

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed — download from [ollama.com](https://ollama.com) (free, open-source)
3. **Qwen2.5 3B model** pulled:

```bash
ollama pull qwen2.5:3b
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/sfc38/local-llm-api.git
cd local-llm-api

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Start Ollama** (in a separate terminal):

```bash
ollama run qwen2.5:3b
```

**Step 2 — Start the API:**

```bash
uvicorn app.main:app --reload
```

**Step 3 — Open the interactive docs:**

```
http://127.0.0.1:8000/docs
```

---

## Example Requests

**Generate a response:**

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what RAG means in simple terms."}'
```

**Response:**

```json
{
  "answer": "RAG means Retrieval-Augmented Generation. It lets an LLM answer using information retrieved from documents."
}
```

---

## Project Structure

```
local-llm-api/
  app/
    __init__.py
    main.py          — FastAPI app and route definitions
    schemas.py       — Pydantic request/response models
    ollama_client.py — HTTP client for Ollama
    services.py      — Business logic per endpoint
  tests/
    test_api.py
  requirements.txt
  README.md
```

---

## Future Improvements

- Temperature and max token parameters on requests
- Model selector (switch between Qwen2.5 variants, Llama 3, etc.)
- Structured logging
- Better error handling and input validation
- Unit and integration tests
- Streamlit frontend for interactive prompting
- Dockerfile for portable deployment
- README screenshots of Swagger UI

---

## License

MIT — free to use, modify, and distribute.
