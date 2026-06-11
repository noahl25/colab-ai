#!/usr/bin/env bash
set -euo pipefail

# ─── colours ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}==> $1${RESET}\n"; }
ok()     { echo -e "${GREEN}✓ $1${RESET}"; }
fail()   { echo -e "${RED}✗ $1${RESET}"; exit 1; }

# ─── pre-flight ─────────────────────────────────────────────────────
banner "Pre-flight checks"

command -v nvidia-smi &>/dev/null || fail "nvidia-smi not found — need a GPU runtime"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ok "GPU detected"

command -v python3 &>/dev/null || fail "python3 not found"
ok "python3 found"

# ─── install system dependencies ────────────────────────────────────
banner "Installing system packages"
if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq wget curl zstd > /dev/null 2>&1
    ok "wget / curl installed"
fi

# ─── install ollama ─────────────────────────────────────────────────
banner "Installing Ollama"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'unknown version'))"
fi

# ─── Python dependencies ────────────────────────────────────────────
banner "Installing Python packages"
pip install --upgrade pip
pip install fastapi uvicorn httpx
ok "Proxy deps installed"

# ─── detect VRAM ────────────────────────────────────────────────────
banner "Detecting GPU memory"
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GIB=$(( VRAM_MIB / 1024 ))
ok "GPU has ~${VRAM_GIB} GiB VRAM"

# ─── choose a model ─────────────────────────────────────────────────
banner "Model selection"
echo ""
echo "Your GPU has ~${VRAM_GIB} GiB VRAM."
echo ""
echo "Recommended models (Ollama library names):"
echo "  1) llama3.3:70b           (70B, ~43 GB, 128K ctx, tools) ← BEST all-round / RAG"
echo "  2) qwen2.5-coder:32b      (32B, ~22 GB, 128K ctx, tools) ← BEST coding (92.7% HumanEval)"
echo "  3) deepseek-r1:32b        (32B, ~20 GB, 128K ctx)        ← BEST reasoning / math"
echo "  4) qwen3-coder:30b        (30B, ~18 GB, 128K ctx, tools) ← BEST agentic coding"
echo "  5) gemma4:26b             (26B, ~17 GB, 128K ctx, tools) ← BEST multimodal / vision"
echo "  6) Enter a custom Ollama model tag"
echo ""
read -rp "Pick [1/2/3/4/5/6] (default 1): " MODEL_CHOICE
case "${MODEL_CHOICE:-1}" in
    1) MODEL="llama3.3:70b" ;;
    2) MODEL="qwen2.5-coder:32b" ;;
    3) MODEL="deepseek-r1:32b" ;;
    4) MODEL="qwen3-coder:30b" ;;
    5) MODEL="gemma4:26b" ;;
    6) read -rp "Model tag (e.g. llama3.3:70b): " MODEL
       [[ -z "$MODEL" ]] && fail "No model specified" ;;
    *) MODEL="llama3.3:70b" ;;
esac
ok "Will serve: ${MODEL}"

# ─── context length ─────────────────────────────────────────────────
echo ""
echo "Ollama defaults to 2048 tokens regardless of the model's native maximum."
echo "Set this to the model's full context length to unlock long-context support."
echo ""
read -rp "Context length [default: 32768]: " NUM_CTX
NUM_CTX="${NUM_CTX:-32768}"
ok "Will use context length: ${NUM_CTX}"

# ─── generate API key ──────────────────────────────────────────────
COLAB_API_KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
export COLAB_API_KEY

# ─── ports ──────────────────────────────────────────────────────────
OLLAMA_PORT=11434
PROXY_PORT=8001

# ─── start Ollama server ────────────────────────────────────────────
banner "Starting Ollama server"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" OLLAMA_NUM_CTX="${NUM_CTX}" ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
ok "Ollama starting (pid ${OLLAMA_PID}) — logs at /tmp/ollama.log"

# wait for Ollama to be ready
banner "Waiting for Ollama to become healthy"
OLLAMA_READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:${OLLAMA_PORT}/ > /dev/null 2>&1; then
        OLLAMA_READY=1
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo ""
        echo -e "${RED}✗ Ollama exited unexpectedly. Last 40 lines of log:${RESET}"
        tail -40 /tmp/ollama.log
        exit 1
    fi
    printf "\r  waiting… %ds" "$i"
    sleep 1
done
echo ""
[[ "$OLLAMA_READY" -eq 1 ]] || fail "Ollama did not start within 60 s — check /tmp/ollama.log"
ok "Ollama is healthy"

# ─── pull model ─────────────────────────────────────────────────────
banner "Pulling model (may take a while on first run)"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" ollama pull "$MODEL"
ok "Model ready: ${MODEL}"

# ─── start FastAPI proxy ───────────────────────────────────────────
banner "Starting auth proxy on port ${PROXY_PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ollama exposes an OpenAI-compatible endpoint at /v1
COLAB_API_KEY="$COLAB_API_KEY" \
VLLM_BASE="http://127.0.0.1:${OLLAMA_PORT}" \
python3 -m uvicorn proxy:app \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --app-dir "$SCRIPT_DIR" \
    > /tmp/proxy.log 2>&1 &
PROXY_PID=$!

sleep 2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    fail "Proxy failed to start — check /tmp/proxy.log"
fi
ok "Auth proxy running (pid ${PROXY_PID})"

# ─── Cloudflare Tunnel ──────────────────────────────────────────────
banner "Setting up Cloudflare Tunnel"

if ! command -v cloudflared &>/dev/null; then
    echo "Downloading cloudflared…"
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
    ok "cloudflared installed"
fi

cloudflared tunnel --url http://127.0.0.1:${PROXY_PORT} \
    > /tmp/cloudflared.log 2>&1 &
CF_PID=$!

# parse the tunnel URL (cloudflared prints it to stderr)
TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1 || true)
    if [[ -n "$TUNNEL_URL" ]]; then break; fi
    sleep 1
done

[[ -n "$TUNNEL_URL" ]] || fail "Could not detect tunnel URL — check /tmp/cloudflared.log"

# ─── summary ───────────────────────────────────────────────────────
banner "All set!"

echo ""
echo -e "${BOLD}Model:${RESET}       ${MODEL}"
echo -e "${BOLD}Tunnel URL:${RESET}  ${TUNNEL_URL}"
echo -e "${BOLD}API Key:${RESET}     ${COLAB_API_KEY}"
echo ""
echo -e "${BOLD}OpenAI-compatible base URL:${RESET}"
echo "  ${TUNNEL_URL}/v1"
echo ""
echo -e "${BOLD}Quick test:${RESET}"
echo "  curl ${TUNNEL_URL}/v1/models \\"
echo "    -H \"Authorization: Bearer ${COLAB_API_KEY}\""
echo ""
echo -e "${BOLD}Use in Cursor / Aider / etc:${RESET}"
echo "  Base URL  ${TUNNEL_URL}/v1"
echo "  API Key   ${COLAB_API_KEY}"
echo ""
echo -e "${BOLD}Logs:${RESET}"
echo "  Ollama      /tmp/ollama.log"
echo "  Proxy       /tmp/proxy.log"
echo "  Cloudflare  /tmp/cloudflared.log"
echo ""
echo -e "${BOLD}Stop everything:${RESET}"
echo "  kill ${OLLAMA_PID} ${PROXY_PID} ${CF_PID}"
echo ""