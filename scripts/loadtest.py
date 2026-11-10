"""Нагрузочный тест на профиле поддержки (triage + draft на каждое обращение).

Два режима:
  closed — N одновременных клиентов без пауз (1/8/32): пропускная способность и задержки;
  open   — пуассоновский поток с заданной частотой, ступенями вверх: максимальная нагрузка,
           при которой p95 полного ответа ещё укладывается в SLA. Именно open-режим отвечает
           на вопрос «сколько запросов в сутки выдержит карта»: в closed-режиме клиенты сами
           притормаживают, когда сервер не успевает, и перегрузку не видно.

    uv run python scripts/loadtest.py --target vllm --name B
    uv run python scripts/loadtest.py --target api --model gpt-6-luna --name api_luna --closed 1,8 --open ""
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import task  # noqa: E402
from bench.client import Result, chat, endpoint  # noqa: E402
from bench.gpu import GpuMonitor  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # консоль Windows по умолчанию cp1251
ROOT = Path(__file__).resolve().parent.parent


def pct(xs: list[float], q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 3) if xs else None


def summarize(results: list[Result], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    out = {
        "requests": len(results), "errors": len(results) - len(ok), "wall_sec": round(wall, 1),
        "req_per_sec": round(len(ok) / wall, 3) if wall else 0,
        "out_tok_per_sec": round(sum(r.tokens_out for r in ok) / wall, 1) if wall else 0,
        "in_tok_per_sec": round(sum(r.tokens_in for r in ok) / wall, 1) if wall else 0,
        "ttft_p50": pct([r.ttft for r in ok if r.ttft is not None], 50),
        "ttft_p95": pct([r.ttft for r in ok if r.ttft is not None], 95),
        "latency_p50": pct([r.latency for r in ok], 50),
        "latency_p95": pct([r.latency for r in ok], 95),
        "by_kind": {},
    }
    for kind in ("triage", "draft"):
        k = [r for r in ok if r.kind == kind]
        if k:
            out["by_kind"][kind] = {
                "n": len(k),
                "tokens_in_avg": round(sum(r.tokens_in for r in k) / len(k), 1),
                "tokens_out_avg": round(sum(r.tokens_out for r in k) / len(k), 1),
                "ttft_p95": pct([r.ttft for r in k if r.ttft is not None], 95),
                "latency_p50": pct([r.latency for r in k], 50),
                "latency_p95": pct([r.latency for r in k], 95),
            }
    errs = [r.error for r in results if not r.ok]
    if errs:
        out["error_sample"] = errs[0]
    return out


async def run_closed(ctx, concurrency: int, duration: float) -> tuple[list[Result], float]:
    client, base, headers, model, reqs, extra, timeout = ctx
    t0, results, i = time.perf_counter(), [], 0

    async def worker():
        nonlocal i
        while time.perf_counter() - t0 < duration:
            req = reqs[i % len(reqs)]
            i += 1
            results.append(await chat(client, base, headers, model, req, t0, timeout, extra))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, time.perf_counter() - t0


async def run_open(ctx, rate: float, duration: float, rng: random.Random) -> tuple[list[Result], float, int]:
    client, base, headers, model, reqs, extra, timeout = ctx
    t0, tasks, i = time.perf_counter(), [], rng.randrange(len(reqs))
    while True:
        await asyncio.sleep(rng.expovariate(rate))
        if time.perf_counter() - t0 >= duration:
            break
        tasks.append(asyncio.create_task(chat(client, base, headers, model, reqs[i % len(reqs)], t0, timeout, extra)))
        i += 1
    results = list(await asyncio.gather(*tasks))
    # пропускная способность считается по окну подачи: «хвост» дообработки в неё не входит
    return results, duration, len(tasks)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["vllm", "ollama", "api"], default="vllm")
    ap.add_argument("--model", default="local")
    ap.add_argument("--name", required=True)
    ap.add_argument("--closed", default="1,8,32", help="число одновременных клиентов, через запятую")
    ap.add_argument("--open", default="1,2,4,6,8,10,12,14,16,18,20,24", help="запросов в секунду, ступени")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--sla", type=float, default=10.0, help="SLA: p95 полного ответа, сек")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base, headers = endpoint(args.target)
    reqs = task.build_requests(task.load_tickets(), task.load_directory())
    if args.target == "api":  # модели с рассуждением тратят выходные токены до ответа — как в eval_triage
        reqs = [{**r, "max_tokens": r["max_tokens"] + 4000} for r in reqs]
    rng = random.Random(args.seed)
    rng.shuffle(reqs)
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "target": args.target, "model": args.model, "sla_sec": args.sla,
              "duration_sec": args.duration, "closed": {}, "open": [], "started": time.strftime("%Y-%m-%d %H:%M")}

    async with httpx.AsyncClient(limits=limits) as client:
        ctx = (client, base, headers, args.model, reqs, None, args.timeout)
        if args.target != "api":  # простой с загруженной моделью (до прогрева): нужен модели стоимости
            with GpuMonitor() as g0:
                w0 = time.time()
                await asyncio.sleep(10)
            report["idle"] = g0.window(w0, time.time())
        warm = await asyncio.gather(*(chat(client, base, headers, args.model, r, time.perf_counter()) for r in reqs[:4]))
        if not any(r.ok for r in warm):
            sys.exit(f"Сервер не отвечает: {warm[0].error}")

        with GpuMonitor() as gpu, open(raw_dir / f"load_{args.name}.jsonl", "w", encoding="utf-8") as raw:
            def dump(stage: str, results: list[Result]) -> None:
                for r in results:
                    raw.write(json.dumps({"stage": stage, **r.row()}, ensure_ascii=False) + "\n")

            for c in [int(x) for x in args.closed.split(",") if x]:
                w0 = time.time()
                results, wall = await run_closed(ctx, c, args.duration)
                s = summarize(results, wall) | (gpu.window(w0, time.time()) if args.target != "api" else {})
                report["closed"][str(c)] = s
                dump(f"closed{c}", results)
                print(f"closed c={c:<3} {s['req_per_sec']:>6} req/s  {s['out_tok_per_sec']:>7} tok/s  "
                      f"TTFT p95 {s['ttft_p95']}  p50 {s['latency_p50']}  p95 {s['latency_p95']}  err {s['errors']}", flush=True)

            for rate in [float(x) for x in args.open.split(",") if x]:
                await asyncio.sleep(3)
                w0 = time.time()
                results, wall, sent = await run_open(ctx, rate, args.duration, rng)
                s = summarize(results, wall) | (gpu.window(w0, time.time()) if args.target != "api" else {})
                s["rate"] = rate
                s["sla_ok"] = bool(s["latency_p95"] is not None and s["latency_p95"] <= args.sla
                                   and s["errors"] <= 0.01 * max(sent, 1))
                report["open"].append(s)
                dump(f"open{rate}", results)
                print(f"open  λ={rate:<4} {s['req_per_sec']:>6} req/s  {s['out_tok_per_sec']:>7} tok/s  "
                      f"TTFT p95 {s['ttft_p95']}  p95 {s['latency_p95']}  err {s['errors']}  "
                      f"{'OK' if s['sla_ok'] else 'SLA НАРУШЕН'}", flush=True)
                if not s["sla_ok"]:
                    break

    passed = [s for s in report["open"] if s["sla_ok"]]
    report["max_rate_within_sla"] = max((s["rate"] for s in passed), default=None)
    (ROOT / "results" / f"load_{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Максимум при p95 ≤ {args.sla} с: {report['max_rate_within_sla']} запросов/с")


if __name__ == "__main__":
    asyncio.run(main())
