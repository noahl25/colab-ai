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

# ─── CUDA library path fix (pip-installed CUDA libs) ────────────────
banner "Configuring CUDA library paths"
NVIDIA_LIBS=$(python3 -c "
import glob, os
try:
    import nvidia
    base = os.path.dirname(nvidia.__path__[0])
    paths = glob.glob(os.path.join(base, 'nvidia', '*', 'lib'))
    print(':'.join(paths))
except ImportError:
    print('')
" 2>/dev/null || true)

if [[ -n "$NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
    ok "Added pip CUDA libs to LD_LIBRARY_PATH"
fi

# also check system CUDA
if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    ok "Added system CUDA to LD_LIBRARY_PATH"
fi

# ─── Hugging Face login ─────────────────────────────────────────────
banner "Hugging Face login"
echo "Some models (Llama, gated Qwen, etc.) need a HF token."
echo "Grab one from https://huggingface.co/settings/tokens"
echo ""
hf auth login

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
echo "Recommended models:"
echo "  1) Qwen/Qwen2.5-72B-Instruct-AWQ     (72B, 4-bit, ~38 GB, 128K context) ← BEST"
echo "  2) Qwen/Qwen2.5-Coder-32B-Instruct   (32B, bf16, ~64 GB, 32K context)"
echo "  3) Qwen/Qwen3-32B                     (32B, bf16, ~64 GB)"
echo "  4) Enter a custom HF model id"
echo ""
read -rp "Pick [1/2/3/4] (default 1): " MODEL_CHOICE
EXTRA_ARGS=""
case "${MODEL_CHOICE:-1}" in
    1) MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
       EXTRA_ARGS="--quantization awq" ;;
    2) MODEL="Qwen/Qwen2.5-Coder-32B-Instruct" ;;
    3) MODEL="Qwen/Qwen3-32B" ;;
    4) read -rp "Model id (org/name): " MODEL
       [[ -z "$MODEL" ]] && fail "No model specified"
       read -rp "Quantization? [none/awq/gptq/fp8] (default none): " QUANT
       if [[ -n "$QUANT" && "$QUANT" != "none" ]]; then
           EXTRA_ARGS="--quantization ${QUANT}"
       fi ;;
    *) MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
       EXTRA_ARGS="--quantization awq" ;;
esac
ok "Will serve: ${MODEL} ${EXTRA_ARGS}"

# ─── configurable context length ────────────────────────────────────
echo ""
echo "Context length is capped by the model's max_position_embeddings."
echo "vLLM will auto-detect the model's native max if you leave this blank."
echo ""
read -rp "Max context length [default: auto-detect from model]: " MAX_CTX

MAX_CTX_ARGS=""
if [[ -n "$MAX_CTX" ]]; then
    MAX_CTX_ARGS="--max-model-len ${MAX_CTX}"
    ok "Will use --max-model-len ${MAX_CTX}"
else
    ok "Will use model's native max context length"
fi

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
    --gpu-memory-utilization 0.92 \
    ${MAX_CTX_ARGS} \
    ${EXTRA_ARGS} \
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
        echo -e "${RED}✗ vLLM exited unexpectedly. Last 40 lines of log:${RESET}"
        tail -40 /tmp/vllm.log
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
