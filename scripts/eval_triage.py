"""Качество на задаче поддержки: 150 размеченных обращений, тот же промпт для своей модели и для API.

Считает долю верных полей (тема, приоритет, маршрут), «всё верно сразу», долю невалидного JSON,
а на черновиках — детерминированные проверки (длина, иероглифы в русском тексте).
Заодно меряет фактические токены на обращение: из них считается стоимость API.

    uv run python scripts/eval_triage.py --target vllm --name B
    uv run python scripts/eval_triage.py --target api --model gpt-6-luna --name api_gpt-6-luna
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import task  # noqa: E402
from bench.client import chat, endpoint  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # консоль Windows по умолчанию cp1251
ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[぀-ヿ㐀-鿿]")


def wilson(k: int, n: int, z: float = 1.96) -> float:
    """Полуширина 95% интервала Вилсона."""
    if not n:
        return 0.0
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["vllm", "ollama", "api"], default="vllm")
    ap.add_argument("--model", default="local")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    base, headers = endpoint(args.target)
    d, tickets = task.load_directory(), task.load_tickets()[: args.limit]
    sem, t0 = asyncio.Semaphore(args.workers), time.perf_counter()
    # API-модели с рассуждением тратят выходные токены до ответа: даём им запас, платим по факту
    slack = 4000 if args.target == "api" else 0

    async with httpx.AsyncClient() as client:
        async def one(t: dict) -> dict:
            async with sem:
                req = {"kind": "triage", "id": t["id"], "messages": task.triage_messages(t, d),
                       "max_tokens": task.TRIAGE_MAX_TOKENS + slack, "temperature": 0.0}
                tr = await chat(client, base, headers, args.model, req, t0, args.timeout)
                if not tr.ok:  # одна повторная попытка на сетевой сбой
                    tr = await chat(client, base, headers, args.model, req, t0, args.timeout)
                pred = task.parse_triage(tr.content, d) if tr.ok else None
                row = {"id": t["id"], "gold_topic": t["topic"], "gold_priority": t["priority"],
                       "segment": t["segment"], "source": t.get("source"), "pred": pred,
                       "raw": tr.content[:300], "triage": tr.row()}
                if not args.no_drafts and t["topic"] != "spam":
                    dreq = {"kind": "draft", "id": t["id"], "messages": task.draft_messages(t),
                            "max_tokens": task.DRAFT_MAX_TOKENS + slack, "temperature": 0.7}
                    dr = await chat(client, base, headers, args.model, dreq, t0, args.timeout)
                    row["draft"], row["draft_meta"] = dr.content, dr.row()
                return row

        rows = await asyncio.gather(*(one(t) for t in tickets))

    n = len(rows)
    hit = {"topic": 0, "priority": 0, "route": 0, "all": 0}
    invalid = 0
    for r in rows:
        p = r["pred"]
        if not p:
            invalid += 1
            continue
        ok_t = p["topic"] == r["gold_topic"]
        ok_p = p["priority"] == r["gold_priority"]
        # маршрут проверяется по ЭТАЛОННОЙ теме: ошибка темы не должна считаться дважды
        ok_r = p["assignee"] in task.candidates(r["gold_topic"], r["segment"], d)
        hit["topic"] += ok_t
        hit["priority"] += ok_p
        hit["route"] += ok_r
        hit["all"] += ok_t and ok_p and ok_r

    drafts = [r for r in rows if r.get("draft")]
    tr_ok = [r["triage"] for r in rows if r["triage"]["ok"]]
    dr_ok = [r["draft_meta"] for r in rows if r.get("draft_meta", {}).get("ok")]
    avg = lambda xs, k: round(sum(x[k] for x in xs) / len(xs), 1) if xs else 0  # noqa: E731
    summary = {
        "name": args.name, "target": args.target, "model": args.model, "n": n,
        "accuracy": {k: round(v / n, 4) for k, v in hit.items()},
        "ci95": {k: round(wilson(v, n), 4) for k, v in hit.items()},
        "invalid_json": invalid,
        "tokens": {"triage_in": avg(tr_ok, "tokens_in"), "triage_out": avg(tr_ok, "tokens_out"),
                   "draft_in": avg(dr_ok, "tokens_in"), "draft_out": avg(dr_ok, "tokens_out")},
        "drafts": {
            "n": len(drafts),
            "failed": sum(1 for r in rows if "draft_meta" in r and not r["draft_meta"]["ok"]),
            "with_cjk": sum(1 for r in drafts if CJK.search(r["draft"])),
            "truncated": sum(1 for r in drafts if r["draft_meta"]["tokens_out"] >= task.DRAFT_MAX_TOKENS + slack),
            "chars_avg": round(sum(len(r["draft"]) for r in drafts) / len(drafts)) if drafts else 0,
        },
        "wall_sec": round(time.perf_counter() - t0, 1),
    }
    out = ROOT / "results"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"quality_{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "raw" / f"quality_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
