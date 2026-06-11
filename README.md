# colab-ai

One-command setup to serve an open LLM on a Colab GPU and expose it as an
OpenAI-compatible API behind Cloudflare Tunnel with API key auth.

## Quick start (Google Colab)

1. Open a Colab notebook with a **GPU runtime** (A100 / L4 / etc.).

2. Run in a notebook cell:

```python
!pip install colab-xterm
%load_ext colabxterm
%xterm
```

3. In the xterm terminal that opens:

```bash
git clone https://github.com/YOUR_USER/colab-ai.git
cd colab-ai
bash setup.sh
```

The script will:

- Install vLLM and dependencies
- Prompt you to log into Hugging Face
- Let you pick a model (Qwen 72B, Qwen3-32B, or custom)
- Start vLLM serving the model
- Start a FastAPI auth proxy
- Open a Cloudflare quick tunnel
- Print the **tunnel URL** and **API key**

## Using the endpoint

Point any OpenAI-compatible client at the printed URL:

| Setting   | Value                                |
|-----------|--------------------------------------|
| Base URL  | `https://<tunnel>.trycloudflare.com/v1` |
| API Key   | *(printed by setup.sh)*             |

### Cursor

Settings → Models → OpenAI API Base → paste the base URL and API key.

### curl

```bash
curl https://<tunnel>.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Logs

| Service    | Log file              |
|------------|-----------------------|
| vLLM       | `/tmp/vllm.log`       |
| Auth proxy | `/tmp/proxy.log`      |
| Cloudflare | `/tmp/cloudflared.log`|

## Stopping

Press **Ctrl-C** in the xterm, or run the `kill` command printed at the end
of setup.
