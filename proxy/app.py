#!/usr/bin/env python3
"""
FastAPI reverse proxy that sits in front of vLLM and enforces
Bearer-token authentication. Every request to /v1/* is validated
then forwarded to the local vLLM server.
"""

import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")
API_KEY = os.environ["COLAB_API_KEY"]

app = FastAPI(title="colab-ai proxy")


def _check_auth(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[len("Bearer "):]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(request: Request, path: str):
    _check_auth(request)

    target = f"{VLLM_BASE}/v1/{path}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "content-length")
    }

    body = await request.body()
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    try:
        upstream = await client.request(
            method=request.method,
            url=target,
            headers=headers,
            content=body,
            params=dict(request.query_params),
        )
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse(
            status_code=502,
            content={"error": "vLLM backend is not reachable"},
        )

    is_stream = "text/event-stream" in upstream.headers.get("content-type", "")

    if is_stream:
        async def _stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            _stream(),
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
        )

    await client.aclose()
    return JSONResponse(
        status_code=upstream.status_code,
        content=upstream.json() if upstream.headers.get("content-type", "").startswith("application/json") else {"raw": upstream.text},
        headers={
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
        },
    )
