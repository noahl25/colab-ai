# colab-ai

One-command setup to serve an open LLM on a Colab GPU and expose it as an
OpenAI-compatible API behind a Cloudflare Tunnel with API-key auth.

Two backends are supported:

- **Ollama** (`scripts/setup-ollama.sh`) — simplest, GGUF models, no HF token.
- **vLLM** (`scripts/setup-vllm.sh`) — higher throughput, HF models / AWQ quants.

## Layout

```
colab-ai/
├── proxy/app.py          FastAPI auth proxy (Bearer token → local backend)
├── scripts/
│   ├── common.sh         shared helpers (preflight, proxy, tunnel, summary)
│   ├── setup-ollama.sh    Ollama backend
│   └── setup-vllm.sh      vLLM backend
└── tests/smoke_test.py   end-to-end smoke test against the tunnel URL
```

## Quick start (Google Colab)

1. Open a Colab notebook with a **GPU runtime** (A100 / L4 / etc.).

2. Run in a notebook cell:

   ```python
   !pip install colab-xterm
   %load_ext colabxterm
   %xterm
   ```

3. In the xterm terminal that opens, clone and run one of the setup scripts:

   ```bash
   git clone https://github.com/YOUR_USER/colab-ai.git
   cd colab-ai
   bash scripts/setup-ollama.sh     # or: bash scripts/setup-vllm.sh
   ```

The script will:

- Run pre-flight checks and install dependencies
- Let you pick a model (or enter a custom one)
- Start the backend serving the model
- Start the FastAPI auth proxy
- Open a Cloudflare quick tunnel
- Print the **tunnel URL** and **API key**

## Using the endpoint

Point any OpenAI-compatible client at the printed URL:

| Setting   | Value                                   |
|-----------|-----------------------------------------|
| Base URL  | `https://<tunnel>.trycloudflare.com/v1` |
| API Key   | *(printed at the end of the script)*    |

### Cursor

Settings → Models → OpenAI API Base → paste the base URL and API key.

### curl

```bash
curl https://<tunnel>.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.3:70b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Smoke test

```bash
python tests/smoke_test.py     # prompts for the base URL and API key
```

## Logs

| Service    | Log file               |
|------------|------------------------|
| Backend    | `/tmp/ollama.log` or `/tmp/vllm.log` |
| Auth proxy | `/tmp/proxy.log`       |
| Cloudflare | `/tmp/cloudflared.log` |

## Stopping

Press **Ctrl-C** in the xterm, or run the `kill` command printed at the end
of the setup script.
