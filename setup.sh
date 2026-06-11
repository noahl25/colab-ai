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
    apt-get update -qq && apt-get install -y -qq wget curl > /dev/null 2>&1
    ok "wget / curl installed"
fi

# ─── Python dependencies ────────────────────────────────────────────
banner "Installing Python packages"
pip install --upgrade pip
pip install vllm huggingface_hub fastapi uvicorn httpx
ok "vLLM + proxy deps installed"

# ─── Hugging Face login ─────────────────────────────────────────────
banner "Hugging Face login"
echo "Some models (Llama, gated Qwen, etc.) need a HF token."
echo "Grab one from https://huggingface.co/settings/tokens"
echo ""
hf auth login

# ─── choose a model ─────────────────────────────────────────────────
banner "Model selection"
DEFAULT_MODEL="Qwen/Qwen2.5-72B-Instruct"
echo "Popular choices for ≈100 GB VRAM:"
echo "  1) Qwen/Qwen2.5-72B-Instruct   (excellent coding, fast)"
echo "  2) Qwen/Qwen3-32B              (newer Qwen family)"
echo "  3) Enter a custom HF model id"
echo ""
read -rp "Pick [1/2/3] (default 1): " MODEL_CHOICE
case "${MODEL_CHOICE:-1}" in
    1) MODEL="Qwen/Qwen2.5-72B-Instruct" ;;
    2) MODEL="Qwen/Qwen3-32B" ;;
    3) read -rp "Model id (org/name): " MODEL
       [[ -z "$MODEL" ]] && fail "No model specified" ;;
    *) MODEL="$DEFAULT_MODEL" ;;
esac
ok "Will serve: ${MODEL}"

# ─── configurable context length ────────────────────────────────────
read -rp "Max context length [default 32768]: " MAX_CTX
MAX_CTX="${MAX_CTX:-32768}"

# ─── generate API key ──────────────────────────────────────────────
COLAB_API_KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
export COLAB_API_KEY

# ─── ports ──────────────────────────────────────────────────────────
VLLM_PORT=8000
PROXY_PORT=8001

# ─── start vLLM ────────────────────────────────────────────────────
banner "Starting vLLM (model download may take a while on first run)"
vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_CTX" \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
ok "vLLM starting (pid ${VLLM_PID}) — logs at /tmp/vllm.log"

# wait for vLLM to be ready (up to 10 min for large downloads)
banner "Waiting for vLLM to become healthy"
VLLM_READY=0
for i in $(seq 1 600); do
    if curl -sf http://127.0.0.1:${VLLM_PORT}/health > /dev/null 2>&1; then
        VLLM_READY=1
        break
    fi
    # check the process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        fail "vLLM exited unexpectedly. Last 30 lines of log:"
        tail -30 /tmp/vllm.log
        exit 1
    fi
    printf "\r  waiting… %ds" "$i"
    sleep 1
done
echo ""
[[ "$VLLM_READY" -eq 1 ]] || fail "vLLM did not start within 600 s — check /tmp/vllm.log"
ok "vLLM is healthy"

# ─── start FastAPI proxy ───────────────────────────────────────────
banner "Starting auth proxy on port ${PROXY_PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COLAB_API_KEY="$COLAB_API_KEY" \
VLLM_BASE="http://127.0.0.1:${VLLM_PORT}" \
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
banner "🎉  All set!"

cat <<EOF

${BOLD}Model:${RESET}       ${MODEL}
${BOLD}Tunnel URL:${RESET}  ${TUNNEL_URL}
${BOLD}API Key:${RESET}     ${COLAB_API_KEY}

${BOLD}OpenAI-compatible base URL:${RESET}
  ${TUNNEL_URL}/v1

${BOLD}Quick test:${RESET}
  curl ${TUNNEL_URL}/v1/models \\
    -H "Authorization: Bearer ${COLAB_API_KEY}"

${BOLD}Use in Cursor / Aider / etc:${RESET}
  Base URL → ${TUNNEL_URL}/v1
  API Key  → ${COLAB_API_KEY}

${BOLD}Logs:${RESET}
  vLLM        → /tmp/vllm.log
  Proxy       → /tmp/proxy.log
  Cloudflare  → /tmp/cloudflared.log

${BOLD}Stop everything:${RESET}
  kill ${VLLM_PID} ${PROXY_PID} ${CF_PID}

EOF

# keep the script alive so the user can Ctrl-C to stop all
wait
