#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# setup-vllm.sh — serve a Hugging Face model with vLLM on a Colab GPU,
# fronted by the auth proxy and exposed via a Cloudflare quick tunnel.
# ─────────────────────────────────────────────────────────────────────

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VLLM_PORT=8000

preflight

# ─── Python dependencies ─────────────────────────────────────────────
banner "Installing Python packages"
pip install --upgrade pip
pip install vllm huggingface_hub fastapi uvicorn httpx
ok "vLLM + proxy deps installed"

# ─── CUDA library path fix (pip-installed CUDA libs) ─────────────────
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

if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    ok "Added system CUDA to LD_LIBRARY_PATH"
fi

# ─── Hugging Face login ──────────────────────────────────────────────
banner "Hugging Face login"
echo "Some models (Llama, gated Qwen, etc.) need a HF token."
echo "Grab one from https://huggingface.co/settings/tokens"
echo ""
hf auth login

detect_vram

# ─── choose a model ─────────────────────────────────────────────────
banner "Model selection"
echo ""
echo "Your GPU has ~${VRAM_GIB} GiB VRAM."
echo "Browse models at https://huggingface.co/models"
echo ""
read -rp "HF model id (org/name): " MODEL
[[ -z "$MODEL" ]] && fail "No model specified"

# AWQ-quantized models need an explicit --quantization flag.
read -rp "Quantization [blank=none, e.g. awq_marlin]: " QUANT
EXTRA_ARGS=""
[[ -n "$QUANT" ]] && EXTRA_ARGS="--quantization ${QUANT}"
ok "Will serve: ${MODEL} ${EXTRA_ARGS}"

# ─── configurable context length ─────────────────────────────────────
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

generate_api_key

# ─── start vLLM ──────────────────────────────────────────────────────
banner "Starting vLLM (model download may take a while on first run)"

# FlashInfer JIT can't detect Blackwell (SM 12.x) on CUDA < 12.9,
# causing a false "requires sm75 or higher" error. Disable it.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL_ALIAS=$(echo "$MODEL" | sed 's|.*/||')

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.92 \
    --served-model-name "$MODEL" "$MODEL_ALIAS" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    ${MAX_CTX_ARGS} \
    ${EXTRA_ARGS} \
    > /tmp/vllm.log 2>&1 &
BACKEND_PID=$!
ok "vLLM starting (pid ${BACKEND_PID}) — logs at /tmp/vllm.log"

# Large downloads can take a while, so allow up to 10 min.
wait_for_health "http://127.0.0.1:${VLLM_PORT}/health" 600 "$BACKEND_PID" /tmp/vllm.log "vLLM"

# ─── proxy + tunnel ──────────────────────────────────────────────────
start_proxy "http://127.0.0.1:${VLLM_PORT}"
start_tunnel

print_summary "$MODEL" "vLLM       " /tmp/vllm.log
