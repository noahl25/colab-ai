#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# setup-ollama.sh — serve an Ollama model on a Colab GPU, fronted by the
# auth proxy and exposed via a Cloudflare quick tunnel.
# ─────────────────────────────────────────────────────────────────────

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OLLAMA_PORT=11434

# ─── preflight (zstd is needed by the ollama installer) ──────────────
preflight zstd

# ─── install ollama ─────────────────────────────────────────────────
banner "Installing Ollama"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'unknown version'))"
fi

# ─── Python dependencies (proxy) ─────────────────────────────────────
banner "Installing Python packages"
pip install --upgrade pip
pip install fastapi uvicorn httpx
ok "Proxy deps installed"

detect_vram

# ─── choose a model ─────────────────────────────────────────────────
banner "Model selection"
echo ""
echo "Your GPU has ~${VRAM_GIB} GiB VRAM."
echo "Browse the model library at https://ollama.com/library"
echo ""
read -rp "Ollama model tag (e.g. llama3.3:70b): " MODEL
[[ -z "$MODEL" ]] && fail "No model specified"
ok "Will serve: ${MODEL}"

# ─── context length (maximum possible) ───────────────────────────────
NUM_CTX=131072
ok "Using maximum context length: ${NUM_CTX}"

generate_api_key

# ─── start Ollama server ─────────────────────────────────────────────
banner "Starting Ollama server"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" OLLAMA_CONTEXT_LENGTH="${NUM_CTX}" \
    ollama serve > /tmp/ollama.log 2>&1 &
BACKEND_PID=$!
ok "Ollama starting (pid ${BACKEND_PID}) — logs at /tmp/ollama.log"

wait_for_health "http://127.0.0.1:${OLLAMA_PORT}/" 60 "$BACKEND_PID" /tmp/ollama.log "Ollama"

# ─── pull model ─────────────────────────────────────────────────────
banner "Pulling model (may take a while on first run)"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" ollama pull "$MODEL"
ok "Model ready: ${MODEL}"

# ─── warm up: load model into VRAM ──────────────────────────────────
banner "Loading model into VRAM"
# Empty prompt + keep_alive:-1 loads weights into GPU memory and keeps them
# resident indefinitely, so the first real request isn't slowed by a cold load.
curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
    -d "{\"model\": \"${MODEL}\", \"keep_alive\": -1}" > /dev/null \
    && ok "Model loaded and pinned in VRAM" \
    || echo -e "${RED}✗ Warm-up request failed — model will load on first request${RESET}"

# ─── proxy + tunnel ──────────────────────────────────────────────────
start_proxy "http://127.0.0.1:${OLLAMA_PORT}"
start_tunnel

print_summary "$MODEL" "Ollama     " /tmp/ollama.log
