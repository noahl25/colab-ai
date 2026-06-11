#!/usr/bin/env python3
"""Quick smoke test for the colab-ai endpoint."""

import sys

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

base_url = input("Base URL (e.g. https://xxx.trycloudflare.com/v1): ").strip().rstrip("/")
if not base_url.endswith("/v1"):
    base_url += "/v1"
api_key = input("API Key: ").strip()

client = OpenAI(base_url=base_url, api_key=api_key)

print("\n--- Models ---")
for m in client.models.list():
    print(f"  {m.id}")

print("\n--- Chat completion (streaming) ---")
stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "Write a Python function that reverses a linked list. Be concise."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()


