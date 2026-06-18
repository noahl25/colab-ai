# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────
# common.sh — shared helpers for the colab-ai setup scripts.
# Source this from setup-ollama.sh / setup-vllm.sh; do not run directly.
# ─────────────────────────────────────────────────────────────────────

# ─── colours ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}==> $1${RESET}\n"; }
ok()     { echo -e "${GREEN}✓ $1${RESET}"; }
fail()   { echo -e "${RED}✗ $1${RESET}"; exit 1; }

# Repo layout, derived from this file's location (scripts/common.sh).
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"

PROXY_PORT=8001

# ─── preflight ───────────────────────────────────────────────────────
# Verify a GPU runtime and python3 are available. Pass extra apt packages
# to install as arguments (wget/curl are always installed).
preflight() {
    banner "Pre-flight checks"
    command -v nvidia-smi &>/dev/null || fail "nvidia-smi not found — need a GPU runtime"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ok "GPU detected"

    command -v python3 &>/dev/null || fail "python3 not found"
    ok "python3 found"

    banner "Installing system packages"
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq wget curl "$@" > /dev/null 2>&1
        ok "system packages installed (wget curl $*)"
    fi
}

# ─── VRAM detection ──────────────────────────────────────────────────
# Sets the global VRAM_GIB.
detect_vram() {
    banner "Detecting GPU memory"
    local mib
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    VRAM_GIB=$(( mib / 1024 ))
    ok "GPU has ~${VRAM_GIB} GiB VRAM"
}

# ─── API key ─────────────────────────────────────────────────────────
# Sets and exports the global COLAB_API_KEY.
generate_api_key() {
    COLAB_API_KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    export COLAB_API_KEY
}

# ─── health wait ─────────────────────────────────────────────────────
# wait_for_health <url> <timeout_s> <pid> <logfile> <name>
# Polls <url> until it returns 2xx, aborting early if <pid> dies.
wait_for_health() {
    local url=$1 timeout=$2 pid=$3 logfile=$4 name=$5
    banner "Waiting for ${name} to become healthy"
    local i
    for ((i = 1; i <= timeout; i++)); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo ""
            ok "${name} is healthy"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo -e "${RED}✗ ${name} exited unexpectedly. Last 40 lines of log:${RESET}"
            tail -40 "$logfile"
            exit 1
        fi
        printf "\r  waiting… %ds" "$i"
        sleep 1
    done
    echo ""
    fail "${name} did not start within ${timeout}s — check ${logfile}"
}

# ─── auth proxy ──────────────────────────────────────────────────────
# start_proxy <backend_base_url>
# Starts the FastAPI auth proxy in front of the backend. Sets PROXY_PID.
start_proxy() {
    local backend_base=$1
    banner "Starting auth proxy on port ${PROXY_PORT}"

    COLAB_API_KEY="$COLAB_API_KEY" \
    BACKEND_BASE="$backend_base" \
    python3 -m uvicorn app:app \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --app-dir "${REPO_ROOT}/proxy" \
        > /tmp/proxy.log 2>&1 &
    PROXY_PID=$!

    sleep 2
    kill -0 "$PROXY_PID" 2>/dev/null || fail "Proxy failed to start — check /tmp/proxy.log"
    ok "Auth proxy running (pid ${PROXY_PID})"
}

# ─── Cloudflare tunnel ───────────────────────────────────────────────
# Starts a Cloudflare quick tunnel to the proxy. Sets CF_PID and TUNNEL_URL.
start_tunnel() {
    banner "Setting up Cloudflare Tunnel"

    if ! command -v cloudflared &>/dev/null; then
        echo "Downloading cloudflared…"
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
            -O /usr/local/bin/cloudflared
        chmod +x /usr/local/bin/cloudflared
        ok "cloudflared installed"
    fi

    cloudflared tunnel --url "http://127.0.0.1:${PROXY_PORT}" > /tmp/cloudflared.log 2>&1 &
    CF_PID=$!

    TUNNEL_URL=""
    local i
    for ((i = 1; i <= 30; i++)); do
        TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1 || true)
        [[ -n "$TUNNEL_URL" ]] && break
        sleep 1
    done
    [[ -n "$TUNNEL_URL" ]] || fail "Could not detect tunnel URL — check /tmp/cloudflared.log"
}

# ─── summary ─────────────────────────────────────────────────────────
# print_summary <model> <backend_name> <backend_logfile>
print_summary() {
    local model=$1 backend_name=$2 backend_log=$3
    banner "All set!"
    echo ""
    echo -e "${BOLD}Model:${RESET}       ${model}"
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
    echo "  ${backend_name}  ${backend_log}"
    echo "  Proxy       /tmp/proxy.log"
    echo "  Cloudflare  /tmp/cloudflared.log"
    echo ""
    echo -e "${BOLD}Stop everything:${RESET}"
    echo "  kill ${BACKEND_PID} ${PROXY_PID} ${CF_PID}"
    echo ""
}
