"""Модель стоимости: свой GPU против API и точка безубыточности.

Единица — ОБРАЩЕНИЕ (triage + черновик), а не запрос к модели: так считает бизнес.

API:   цена обращения = фактические токены модели на eval-наборе × ОФИЦИАЛЬНЫЙ прайс провайдера.
Свой:  в месяц = амортизация + электричество (+ сопровождение в полном сценарии);
       мощность одной карты = максимум запросов/с при p95 ≤ SLA из нагрузочного теста,
       делённый на коэффициент пика (сервер нужен под пиковый час, а не под средний).
       Когда поток превышает мощность — докупается следующая карта (ступенька на графике).

    uv run python scripts/cost_model.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # консоль Windows по умолчанию cp1251
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
P = yaml.safe_load((ROOT / "configs/prices.yaml").read_text(encoding="utf-8"))
VARIANTS = yaml.safe_load((ROOT / "configs/variants.yaml").read_text(encoding="utf-8"))["variants"]
FX = P["fx"]["usd_kzt"]
HOURS_MONTH = 730.0
DAYS_MONTH = 30.4


def load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def api_cost_per_ticket(model: str, tok: dict, spam_share: float) -> float:
    """USD за одно обращение. Черновик пишется не на спам, отсюда (1 − spam_share)."""
    pr = P["api"][model]
    tin = tok["triage_in"] + (1 - spam_share) * tok["draft_in"]
    tout = tok["triage_out"] + (1 - spam_share) * tok["draft_out"]
    return (tin * pr["input"] + tout * pr["output"]) / 1e6


def own_capacity_per_day(load_res: dict, reqs_per_ticket: float) -> float:
    """Обращений в сутки на одну карту при соблюдении SLA в пиковый час."""
    rate = load_res.get("max_rate_within_sla") or 0
    return rate / reqs_per_ticket / P["load"]["peak_factor"] * 86400


def own_monthly(tickets_day: float, cap_day: float, power: dict, scenario: str) -> tuple[float, int]:
    """USD в месяц и число карт. scenario: card | full."""
    g, e, o = P["gpu"], P["electricity"], P["ops"]
    n = max(1, math.ceil(tickets_day / cap_day)) if cap_day else 1
    months = g["amortization_years"] * 12
    capex = g["price_kzt"] * (1 - g["residual_share"])
    if scenario == "full":
        capex += g["host_price_kzt"] * (1 - g["residual_share"])  # один хост на каждую карту
    amort = n * capex / months / FX
    # карта работает под нагрузкой долю времени util, остальное простаивает с загруженной моделью
    util = min(1.0, tickets_day / (n * cap_day * P["load"]["peak_factor"])) if cap_day else 0
    watts = power["idle_w"] + (power["load_w"] - power["idle_w"]) * util + g["host_overhead_w"]
    energy = n * watts / 1000 * HOURS_MONTH * e["kzt_per_kwh"] / FX
    admin = o["admin_hours_per_month"] * o["admin_rate_usd"] if scenario == "full" else 0
    return amort + energy + admin, n


def breakeven(api_per_ticket: float, cap_day: float, power: dict, scenario: str) -> float | None:
    """Минимум обращений в сутки, начиная с которого свой GPU дешевле API."""
    for t in np.unique(np.round(np.logspace(0, 7, 4000))):
        own, _ = own_monthly(t, cap_day, power, scenario)
        if own < api_per_ticket * t * DAYS_MONTH:
            return float(t)
    return None


INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
OWN_COLORS = ["#2a78d6", "#0d366b"]  # свой GPU — один оттенок: основной вариант и тёмный штрихпунктир
API_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # API по убыванию цены
CHART_VARIANTS = ["B", "D"]  # самый быстрый 7B и самый качественный 14B


def fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def main() -> None:
    tickets = [json.loads(l) for l in open(ROOT / "data/tickets.jsonl", encoding="utf-8")]
    spam = sum(t["topic"] == "spam" for t in tickets) / len(tickets)
    rpt = 1 + (1 - spam)

    apis = {}
    for model in P["api"]:
        q = load(f"quality_api_{model}.json")
        if q:
            apis[model] = {"per_ticket": api_cost_per_ticket(model, q["tokens"], spam), "quality": q, "measured": True}
    # модели без прогона (недоступны на ключе): только прайс, токены — как у самой экономной из измеренных
    if apis:
        ref = min(apis, key=lambda m: sum(apis[m]["quality"]["tokens"].values()))
        for model in P["api"]:
            if model not in apis:
                apis[model] = {"per_ticket": api_cost_per_ticket(model, apis[ref]["quality"]["tokens"], spam),
                               "quality": None, "measured": False, "tokens_from": ref}
    own = {}
    for key in VARIANTS:
        l, q = load(f"load_{key}.json"), load(f"quality_{key}.json")
        if not l:
            continue
        stages = list(l["closed"].values()) + l["open"]
        power = {"idle_w": l.get("idle", {}).get("power_w_avg", 15.0),
                 "load_w": max((s.get("power_w_avg") or 0) for s in stages)}
        own[key] = {"cap_day": own_capacity_per_day(l, rpt), "power": power, "load": l, "quality": q}
    if not apis or not own:
        sys.exit("Нет результатов: сначала loadtest.py и eval_triage.py")

    out = {"as_of": P["as_of"], "usd_kzt": FX, "spam_share": round(spam, 3), "requests_per_ticket": round(rpt, 3),
           "api_usd_per_1000_tickets": {m: round(a["per_ticket"] * 1000, 4) for m, a in apis.items()},
           "api_not_measured": {m: a["tokens_from"] for m, a in apis.items() if not a["measured"]},
           "own": {}}
    for key, o in own.items():
        row = {"max_rate_within_sla": o["load"].get("max_rate_within_sla"),
               "capacity_tickets_per_day": round(o["cap_day"]), **o["power"], "scenarios": {}}
        for sc in ("card", "full"):
            at_cap, _ = own_monthly(o["cap_day"], o["cap_day"], o["power"], sc)
            row["scenarios"][sc] = {
                "usd_month_idle": round(own_monthly(1, o["cap_day"], o["power"], sc)[0], 2),
                "usd_month_at_capacity": round(at_cap, 2),
                "usd_per_1000_tickets_at_capacity": round(at_cap / (o["cap_day"] * DAYS_MONTH) * 1000, 4) if o["cap_day"] else None,
                "breakeven_tickets_per_day": {m: breakeven(a["per_ticket"], o["cap_day"], o["power"], sc)
                                              for m, a in apis.items()},
            }
        out["own"][key] = row
    (RES / "cost_model.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # --- график первого экрана: стоимость в месяц от числа обращений в сутки
    shown = [k for k in CHART_VARIANTS if k in own] or [max(own, key=lambda k: own[k]["cap_day"])]
    main_key = shown[0]
    xs = np.logspace(1, 6.3, 900)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7", "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # API: модели с одинаковой ценой обращения рисуются одной линией
    groups: list[tuple[float, list[str]]] = []
    for m, a in sorted(apis.items(), key=lambda kv: -kv[1]["per_ticket"]):
        if groups and abs(groups[-1][0] - a["per_ticket"]) / a["per_ticket"] < 0.01:
            groups[-1][1].append(m)
        else:
            groups.append((a["per_ticket"], [m]))
    o = own[main_key]
    for (price, models), c in zip(groups, API_COLORS):
        names = " / ".join(m + ("" if apis[m]["measured"] else "*") for m in models)
        ax.plot(xs, price * xs * DAYS_MONTH, color=c, lw=2, label=f"API {names} — ${price * 1000:.2f} за 1000 обращений")
        be = breakeven(price, o["cap_day"], o["power"], "card")
        if be and be < xs[-1]:
            y = price * be * DAYS_MONTH
            ax.scatter([be], [y], color=c, zorder=6, s=70, edgecolors=SURFACE, linewidths=2)
            above = c in API_COLORS[::2]  # подписи через одну — над линией, чтобы не слипались
            ax.annotate(f"{fmt(be)} в сутки", (be, y), textcoords="offset points",
                        xytext=(-8, 9) if above else (8, -16), ha="right" if above else "left",
                        fontsize=9.5, color=INK, fontweight="bold")
    for key, c in zip(shown, OWN_COLORS):
        v = own[key]
        ax.plot(xs, [own_monthly(x, v["cap_day"], v["power"], "card")[0] for x in xs], color=c, lw=2.8 if key == main_key else 2,
                ls="-" if key == main_key else (0, (6, 2, 1.5, 2)), zorder=5,
                label=f"Свой GPU, {VARIANTS[key]['title']} — карта + электричество (1 карта ≤ {fmt(v['cap_day'])}/сут)")
    ax.plot(xs, [own_monthly(x, o["cap_day"], o["power"], "full")[0] for x in xs], color=OWN_COLORS[0], lw=1.6,
            ls=(0, (5, 3)), zorder=4, label=f"Свой GPU, {VARIANTS[main_key]['title']} — плюс сервер и 4 ч админа в месяц")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(1, 3e4)
    ticks_x = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    ax.set_xticks(ticks_x, [fmt(t) for t in ticks_x])
    ticks_y = [1, 10, 100, 1_000, 10_000]
    ax.set_yticks(ticks_y, [f"${fmt(t)}" for t in ticks_y])
    ax.minorticks_off()
    ax.set_xlabel("обращений в сутки (одно обращение = триаж + черновик ответа)")
    ax.set_ylabel("стоимость в месяц")
    ax.set_title(f"Своя LLM на {P['gpu']['name']} против API: где пересекаются затраты",
                 loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(fontsize=8.6, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(INK2)
    note = f"Точки — безубыточность для «{VARIANTS[main_key]['title']}», сценарий «карта + электричество». Цены API — официальные прайсы на {P['as_of']}."
    if any(not a["measured"] for a in apis.values()):
        note += "\n* модель недоступна на ключе: линия по прайсу, качество не измерялось."
    fig.text(0.012, 0.012, note, fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    (ROOT / "report").mkdir(exist_ok=True)
    fig.savefig(ROOT / "report" / "breakeven.png", facecolor=SURFACE)
    print("график: report/breakeven.png")


if __name__ == "__main__":
    main()
