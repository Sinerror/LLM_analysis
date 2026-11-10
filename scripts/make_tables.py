"""Собирает таблицы отчёта из results/*.json -> report/tables.md.

    uv run python scripts/make_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
VARIANTS = yaml.safe_load((ROOT / "configs/variants.yaml").read_text(encoding="utf-8"))["variants"]
PRICES = yaml.safe_load((ROOT / "configs/prices.yaml").read_text(encoding="utf-8"))
FX = PRICES["fx"]["usd_kzt"]


def load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}".replace(".", ",")


def num(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def main() -> None:
    cost = load("cost_model.json") or {}
    out: list[str] = []

    out += ["### Скорость и память (одна RTX 4060 Ti 16 ГБ)", "",
            "| вариант | веса, ГБ | ток/с, 1 клиент | ток/с, 32 клиента | p95 при 32, с | TTFT p95 при 32, с "
            "| макс. запросов/с при p95 ≤ 10 с | обращений в сутки на карту* | Вт под нагрузкой |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, v in VARIANTS.items():
        l = load(f"load_{k}.json")
        if not l:
            continue
        c1, c32 = l["closed"].get("1", {}), l["closed"].get("32", {})
        cap = cost.get("own", {}).get(k, {}).get("capacity_tickets_per_day")
        watts = max((s.get("power_w_avg") or 0) for s in list(l["closed"].values()) + l["open"])
        out.append(f"| **{k}** {v['title']} | {num(v.get('weights_gb'))} | {num(c1.get('out_tok_per_sec'), 0)} "
                   f"| {num(c32.get('out_tok_per_sec'), 0)} | {num(c32.get('latency_p95'))} | {num(c32.get('ttft_p95'), 2)} "
                   f"| {num(l.get('max_rate_within_sla'))} | {num(cap, 0)} | {num(watts, 0)} |")
    out += ["", "\\* с запасом на пик: средний поток = максимум / "
            f"{num(PRICES['load']['peak_factor'], 0)}; одно обращение = {num(cost.get('requests_per_ticket'), 2)} запроса к модели.", ""]

    out += ["### Качество: поддержка (150 размеченных обращений) и DocCheck (56 заложенных ошибок)", "",
            "| модель | тема | приоритет | маршрут | всё верно | ±95% | черновики с иероглифами | DocCheck: найдено ошибок | DocCheck: ложные срабатывания | $ за 1000 обращений |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(f"API {m}", f"quality_api_{m}.json", None, m) for m in PRICES["api"]]
    rows += [(f"**{k}** {v['title']}", f"quality_{k}.json", f"doccheck_{k}.json", k) for k, v in VARIANTS.items()]
    ref = PRICES.get("doccheck_reference")
    for title, qf, df, key in rows:
        q = load(qf)
        if not q:
            if title.startswith("API"):
                price = cost.get("api_usd_per_1000_tickets", {}).get(key)
                out.append(f"| {title} | не измерено | | | | | | | | ≈ {num(price, 2)} (оценка по прайсу) |")
            continue
        a, d = q["accuracy"], q["drafts"]
        dc = (load(df) or {}).get("summary") if df else (ref if ref and ref.get("analyst") == key else None)
        if title.startswith("API"):
            price = num(cost.get("api_usd_per_1000_tickets", {}).get(key), 2)
        else:
            sc = cost.get("own", {}).get(key, {}).get("scenarios", {}).get("card", {})
            price = f"{num(sc.get('usd_per_1000_tickets_at_capacity'), 3)} при полной загрузке"
        out.append(f"| {title} | {pct(a['topic'])} | {pct(a['priority'])} | {pct(a['route'])} | **{pct(a['all'])}** "
                   f"| {pct(q['ci95']['all'])} | {d['with_cjk']} из {d['n']} "
                   f"| {pct(dc['recall_strict']) if dc else '—'} | {pct(dc['fp_strict_clean']) if dc else '—'} | {price} |")
    out.append("")

    out += ["### Задержка API на том же профиле (через роутер, из Казахстана)", "",
            "| модель | клиентов | p50, с | p95, с | TTFT p95, с | ошибок |", "|---|---|---|---|---|---|"]
    for m in PRICES["api"]:
        l = load(f"load_api_{m}.json")
        for c, s in (l or {}).get("closed", {}).items():
            out.append(f"| {m} | {c} | {num(s['latency_p50'])} | {num(s['latency_p95'])} | {num(s['ttft_p95'])} "
                       f"| {s['errors']} из {s['requests']} |")
    out.append("")

    if cost:
        out += ["### Точка безубыточности, обращений в сутки", "",
                "| свой вариант | сценарий | свой GPU, $/мес (₸) | " + " | ".join(f"против {m}" for m in cost["api_usd_per_1000_tickets"]) + " |",
                "|---|---|---|" + "---|" * len(cost["api_usd_per_1000_tickets"])]
        names = {"card": "карта + электричество", "full": "+ сервер и админ"}
        for k, o in cost["own"].items():
            for sc, s in o["scenarios"].items():
                usd = s["usd_month_idle"]
                out.append(f"| **{k}** | {names[sc]} | {num(usd, 0)} ({num(usd * FX, 0)}) | "
                           + " | ".join(num(s["breakeven_tickets_per_day"][m], 0) for m in cost["api_usd_per_1000_tickets"]) + " |")
        out.append("")

    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
