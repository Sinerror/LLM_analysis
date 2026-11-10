"""Потоковый клиент OpenAI-совместимого API с замером TTFT и токенов."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

TARGETS = {
    "vllm": ("VLLM_BASE_URL", "http://127.0.0.1:8000/v1", None),
    "ollama": ("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1", None),
    "api": ("LLM_BASE_URL", "https://router.cheap/v1", "LLM_API_KEY"),
}


def endpoint(target: str) -> tuple[str, dict]:
    env, default, key_env = TARGETS[target]
    base = os.getenv(env, default).rstrip("/")
    key = os.getenv(key_env) if key_env else None
    if key_env and not key:
        raise SystemExit(f"В .env не задан {key_env}")
    return base, ({"Authorization": f"Bearer {key}"} if key else {})


@dataclass
class Result:
    kind: str
    id: str
    ok: bool
    t_start: float          # сек от начала прогона
    ttft: float | None      # до первого токена
    latency: float          # до конца ответа
    tokens_in: int
    tokens_out: int
    content: str = ""
    error: str = ""

    def row(self, with_content: bool = False) -> dict:
        d = asdict(self)
        if not with_content:
            d.pop("content")
        return d


async def chat(client: httpx.AsyncClient, base: str, headers: dict, model: str, req: dict,
               t0: float, timeout: float = 120.0, extra: dict | None = None) -> Result:
    body = {"model": model, "messages": req["messages"], "max_tokens": req["max_tokens"],
            "temperature": req["temperature"], "stream": True,
            "stream_options": {"include_usage": True}, **(extra or {})}
    started = time.perf_counter()
    ttft, parts, tin, tout = None, [], 0, 0
    try:
        async with client.stream("POST", f"{base}/chat/completions", headers=headers, json=body,
                                 timeout=timeout) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode("utf-8", "replace")[:200]
                raise RuntimeError(f"HTTP {r.status_code}: {text}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    tin = chunk["usage"].get("prompt_tokens", 0) or 0
                    tout = chunk["usage"].get("completion_tokens", 0) or 0
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        parts.append(delta)
        content = "".join(parts)
        if not content:
            raise RuntimeError("пустой ответ")
        return Result(req["kind"], req["id"], True, started - t0, ttft, time.perf_counter() - started,
                      tin, tout, content)
    except Exception as e:  # noqa: BLE001
        return Result(req["kind"], req["id"], False, started - t0, ttft, time.perf_counter() - started,
                      tin, tout, "".join(parts), f"{type(e).__name__}: {str(e)[:160]}")
