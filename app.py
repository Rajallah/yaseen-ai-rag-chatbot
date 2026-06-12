"""
Yaseen AI — a RAG-grounded personal chatbot served by a local LLM.

A Flask backend that loads a quantized Mistral-7B-Instruct model with
llama-cpp-python, retrieves relevant facts from a plain-text knowledge
base, and answers in first person as Yaseen's "digital twin".

Author: Yaseen A. Naser (github.com/Rajallah)
"""

from __future__ import annotations

import logging
import os
import re
import threading

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from llama_cpp import Llama

# --------------------------------------------------------------------------
# Configuration (override any of these with environment variables)
# --------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
)
KNOWLEDGE_FILE = os.environ.get(
    "KNOWLEDGE_FILE", os.path.join(BASE_DIR, "data", "persona.txt")
)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = offload all layers; 0 = CPU only
N_CTX = int(os.environ.get("N_CTX", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.4"))
APP_PORT = int(os.environ.get("PORT", "5000"))

MAX_HISTORY_TURNS = 8        # user+assistant messages kept per request
MAX_MESSAGE_CHARS = 2000     # reject absurdly long inputs
RETRIEVER_TOP_K = 4          # facts injected into the prompt per turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("yaseen-ai")

# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------
def load_knowledge(path: str) -> list[str]:
    """Load the knowledge base: one factual statement per line. Lines starting
    with '#' are treated as comments."""
    if not os.path.exists(path):
        log.warning("Knowledge file not found at %s — answers will not be grounded.", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        statements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    log.info("Loaded %d knowledge statements from %s", len(statements), path)
    return statements


KNOWLEDGE: list[str] = load_knowledge(KNOWLEDGE_FILE)

# --------------------------------------------------------------------------
# Retriever (keyword overlap with stopword filtering)
# --------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "she", "so", "tell", "that", "the", "their", "them", "they", "this",
    "to", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "about", "am", "im",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, minus stopwords, with naive plural stripping
    so e.g. 'certifications' matches 'certification'."""
    words = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def retrieve(query: str, statements: list[str], top_k: int = RETRIEVER_TOP_K) -> list[str]:
    """Return the top-k statements sharing the most meaningful words with the query.

    A deliberately simple lexical retriever: it needs no extra models or
    indexes, which keeps the whole app runnable on one machine. The trade-off
    (documented in the README) is that it misses paraphrases — swapping in a
    sentence-transformer embedding index is the planned upgrade.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored: list[tuple[int, str]] = []
    for stmt in statements:
        stmt_words = _tokenize(stmt)
        overlap = query_words & stmt_words
        if not overlap:
            continue
        # Longer (rarer) words count slightly more than short common ones.
        score = sum(min(len(w), 8) for w in overlap)
        scored.append((score, stmt))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [stmt for _, stmt in scored[:top_k]]


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
def load_llm() -> Llama | None:
    if not os.path.exists(MODEL_PATH):
        log.error(
            "Model file not found at %s. Download the GGUF model first — see README.md.",
            MODEL_PATH,
        )
        return None
    try:
        model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            chat_format="mistral-instruct",
            verbose=False,
        )
        log.info(
            "LLM loaded (%s | n_ctx=%d | n_gpu_layers=%d)",
            os.path.basename(MODEL_PATH), N_CTX, N_GPU_LAYERS,
        )
        return model
    except Exception:
        log.exception("Could not load the LLM.")
        return None


llm = load_llm()
_llm_lock = threading.Lock()  # llama.cpp inference is not thread-safe

# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Yaseen AI, the digital twin of Yaseen A. Naser — a computer "
    "science student from Bahrain. Speak in the first person ('I', 'my') as "
    "Yaseen, with a friendly, direct, and professional tone.\n\n"
    "Grounding rules:\n"
    "1. When asked anything about Yaseen, rely ONLY on the facts listed under "
    "'Verified facts about me'. Never invent, infer, or embellish personal details.\n"
    "2. If the answer is not covered by those facts, say: 'That's not something "
    "I have on file — feel free to ask me about my studies, projects, or interests.'\n"
    "3. If someone asks whether you are an AI, be honest: you are an AI "
    "assistant that represents Yaseen, built by Yaseen as a RAG project.\n"
    "4. Keep answers concise. Refuse harmful, offensive, or inappropriate "
    "requests politely.\n"
    "5. For general questions not about Yaseen (e.g. coding help), answer "
    "normally and helpfully."
)


def build_messages(user_message: str, history: list[dict], facts: list[str]) -> list[dict]:
    """Assemble the chat-completion message list for the current turn."""
    system = SYSTEM_PROMPT
    if facts:
        system += "\n\nVerified facts about me:\n" + "\n".join(f"- {f}" for f in facts)

    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    messages.append({"role": "user", "content": user_message})
    return messages


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if llm else "model_not_loaded",
            "model": os.path.basename(MODEL_PATH),
            "knowledge_statements": len(KNOWLEDGE),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    if llm is None:
        return jsonify({"error": "The language model is not loaded on the server."}), 503

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_CHARS} characters)."}), 400
    if not isinstance(history, list):
        history = []

    facts = retrieve(user_message, KNOWLEDGE)
    messages = build_messages(user_message, history, facts)
    log.info("Chat turn | %d history msgs | %d facts retrieved", len(history), len(facts))

    try:
        with _llm_lock:
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                repeat_penalty=1.1,
            )
        response_text = output["choices"][0]["message"]["content"].strip()
    except Exception:
        log.exception("LLM generation failed.")
        return jsonify({"error": "Error generating a response."}), 500

    return jsonify({"response": response_text})


if __name__ == "__main__":
    log.info("Serving frontend from %s", FRONTEND_DIR)
    log.info("Starting on http://localhost:%d", APP_PORT)
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
