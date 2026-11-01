"""Minimal OpenAI-compatible chat client (Meta Model API, Fireworks, LM Studio, Ollama, vLLM...). Standard library only.

Settings come from .env / the environment: META_API_KEY, META_API_BASE, MODEL_API_MODEL, REASONING_EFFORT.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into the environment (never overrides variables already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


class ChatError(RuntimeError):
    def __init__(self, msg, info=None):
        super().__init__(msg)
        self.info = info or {}


def chat(messages, *, base_url, model, api_key, max_tokens=8192, reasoning_effort=None, temperature=0.2, timeout=300):
    """One chat completion. Returns (text, finish_reason)."""
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {})}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            choice = json.loads(resp.read())["choices"][0]
    except urllib.error.HTTPError as e:
        raise ChatError(f"HTTP {e.code}", {"http_body": e.read()[:500].decode(errors="replace")})
    msg = choice.get("message") or {}
    return msg.get("content") or "", choice.get("finish_reason")


def last_json_object(text: str, must_have=()) -> dict:
    """The last JSON object in a reply (code fences and prose allowed) that has all keys in `must_have`."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    dec, found = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in must_have):
            found = obj
    if found is None:
        raise ValueError("no JSON object in reply")
    return found
