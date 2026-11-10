"""Поднимает один вариант из configs/variants.yaml и ждёт готовности.

    uv run python scripts/serve.py B        # vLLM с AWQ
    uv run python scripts/serve.py E        # Ollama
    uv run python scripts/serve.py down
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # консоль Windows по умолчанию cp1251
ROOT = Path(__file__).resolve().parent.parent


def sh(*cmd: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}, check=check)


def down() -> None:
    sh("docker", "compose", "--profile", "ollama", "down", check=False)


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "B"
    if key == "down":
        return down()
    v = yaml.safe_load((ROOT / "configs/variants.yaml").read_text(encoding="utf-8"))["variants"][key]
    down()
    t0 = time.time()
    if v["engine"] == "ollama":
        sh("docker", "compose", "--profile", "ollama", "up", "-d", "ollama")
        time.sleep(3)
        sh("docker", "exec", "ollama", "ollama", "pull", v["model"])
        url, ready = "http://127.0.0.1:11434/v1/models", lambda r: r.status_code == 200
    else:
        sh("docker", "compose", "up", "-d", "vllm", env={
            "MODEL": v["model"], "MAX_LEN": str(v["max_model_len"]),
            "GPU_UTIL": str(v.get("gpu_util", 0.90)), "EXTRA": v.get("extra", "")})
        url, ready = "http://127.0.0.1:8000/health", lambda r: r.status_code == 200
    deadline = time.time() + 3600
    while time.time() < deadline:
        try:
            if ready(httpx.get(url, timeout=3)):
                break
        except httpx.HTTPError:
            pass
        if v["engine"] == "vllm":
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", "vllm"],
                                   capture_output=True, text=True).stdout.strip()
            if state == "exited":
                sh("docker", "logs", "--tail", "40", "vllm", check=False)
                sys.exit("vLLM завершился при старте — см. лог выше")
        time.sleep(3)
    else:
        sys.exit("Сервер не поднялся за час")
    took = round(time.time() - t0)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"serve_{key}.json").write_text(
        json.dumps({"variant": key, **v, "startup_sec": took}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{key} ({v['title']}) готов за {took} с")


if __name__ == "__main__":
    main()
