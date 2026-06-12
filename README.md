# Yaseen AI — Local RAG Chatbot

A personalized, Retrieval-Augmented Generation (RAG) chatbot that runs **entirely on local hardware**. It serves a custom web UI from a Flask backend, retrieves verified facts from a plain-text knowledge base, and generates grounded, first-person answers with a quantized **Mistral-7B-Instruct** model via `llama-cpp-python` (with optional NVIDIA GPU acceleration).

Built by [Yaseen A. Naser](https://www.linkedin.com/in/yaseenanaser) as a hands-on exploration of LLM deployment, RAG, prompt engineering, and full-stack development. Originally created as a school AI/ML project and later cleaned up for public release.

## How it works

```
Browser (HTML/CSS/JS)
      │  POST /chat  { message, history }
      ▼
Flask backend (app.py)
      │
      ├── 1. Retriever — keyword-scores every statement in data/persona.txt
      │       and selects the top matches for the user's question
      │
      ├── 2. Prompt builder — system prompt + retrieved facts + trimmed
      │       chat history + the new message (Mistral chat template)
      │
      └── 3. llama-cpp-python — runs Mistral-7B-Instruct-v0.2 (GGUF, Q4_K_M)
              locally, optionally offloaded to the GPU
      │
      ▼
JSON response → rendered in the chat UI
```

The grounding rule is enforced in the system prompt: anything about Yaseen must come **only** from the retrieved facts; if the knowledge base doesn't cover it, the bot says so instead of guessing. General questions (e.g. coding help) are answered normally.

## Features

- **100% local inference** — no API keys, no data leaves your machine
- **RAG grounding** over a simple, editable one-fact-per-line knowledge base
- **GPU acceleration** through llama.cpp CUDA offloading (configurable, falls back to CPU)
- **Proper Mistral chat templating** via `create_chat_completion` (no hand-rolled `[INST]` strings)
- **Clean, responsive chat UI** with suggestion chips, typing indicator, and a live server-status dot
- **Configurable via environment variables** — model path, context size, temperature, port, GPU layers
- `/health` endpoint for quick diagnostics

## Getting started

### 1. Requirements

- Python 3.10+
- ~5 GB of disk space for the model
- Optional: an NVIDIA GPU with CUDA for fast inference (the project was developed on an RTX 3060 Ti; CPU-only works but is slow)

### 2. Clone and install

```bash
git clone https://github.com/Rajallah/yaseen-ai-rag-chatbot.git
cd yaseen-ai-rag-chatbot

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU build (optional but recommended):** the default `pip install llama-cpp-python` is CPU-only. For NVIDIA CUDA acceleration, install the CUDA Toolkit and build with:
>
> ```bash
> # Windows (PowerShell)
> $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --force-reinstall --no-cache-dir
> # Linux
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
> ```
>
> See the [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python) for prebuilt wheels and other backends (Metal, ROCm, Vulkan).

### 3. Download the model

The model is **not** included in this repository (~4.4 GB). Download the GGUF file and place it in `models/`:

```bash
# from the repo root
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
    mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --local-dir models
```

Or download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` manually from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) into the `models/` folder.

### 4. Run

```bash
python app.py
```

Open <http://localhost:5000> and start chatting.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Path to any GGUF chat model |
| `KNOWLEDGE_FILE` | `data/persona.txt` | Knowledge base (one fact per line, `#` = comment) |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to GPU (`-1` = all, `0` = CPU only) |
| `N_CTX` | `4096` | Context window in tokens |
| `MAX_TOKENS` | `400` | Max tokens per response |
| `TEMPERATURE` | `0.4` | Sampling temperature (lower = more factual) |
| `PORT` | `5000` | Server port |

Example: `N_GPU_LAYERS=20 PORT=8080 python app.py`

## Make it your own twin

Everything persona-specific lives in two places:

1. **`data/persona.txt`** — replace the facts with your own. One statement per line, written in the first person. Only put information you're comfortable making public.
2. **`SYSTEM_PROMPT` in `app.py`** — swap the name and adjust the tone rules.

## Project structure

```
.
├── app.py               # Flask backend: retriever, prompting, LLM inference
├── data/
│   └── persona.txt      # RAG knowledge base (one fact per line)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/              # place the GGUF model here (git-ignored)
├── requirements.txt
└── README.md
```

## Limitations

- **Lexical retriever:** retrieval is keyword-overlap based, so heavy paraphrasing can miss relevant facts. It's intentionally dependency-free; a semantic upgrade is on the roadmap.
- **No streaming:** responses arrive in one piece rather than token by token.
- **Single-worker inference:** requests are serialized with a lock because llama.cpp inference isn't thread-safe; this is a personal-scale app, not a multi-tenant service.
- **Persona drift:** like any LLM persona, very long or adversarial conversations can pull the model off its grounding rules.

## Roadmap

- Semantic retrieval with `sentence-transformers` embeddings
- Token streaming to the frontend (Server-Sent Events)
- Conversation summarization for long chats
- Dockerfile for one-command setup
