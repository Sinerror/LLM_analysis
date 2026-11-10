"""Задача поддержки из Битрикс-кейса: те же промпты, что у baseline «LLM на каждом обращении».

На одно обращение приходится два запроса к модели:
  triage — длинный общий префикс (категории, политика, справочник) + короткий JSON в ответ;
  draft  — короткий вход + черновик ответа клиенту на 3–6 предложений.
Это и есть профиль нагрузки: длины промптов и ответов не придуманы, а получаются из данных.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

TOPICS = {
    "payment": "Оплата: проблемы с деньгами по заказу, который клиент НЕ возвращает — списали дважды, "
               "платёж не прошёл, деньги списаны, а заказа нет, нужен счёт/акт/закрывающие документы, "
               "не применился промокод или бонусы, рассрочка.",
    "delivery": "Доставка: где заказ, сроки, трек не обновляется, курьер не приехал или опоздал, перенос, "
                "смена адреса, привезли не тот товар или не всё (ошибка комплектации), помята упаковка "
                "при целом товаре.",
    "defect": "Брак: товар неисправен или повреждён — не включается, сломался через неделю, трещина, "
              "внутри не хватает детали, странный шум или запах. Даже если клиент хочет вернуть деньги — "
              "причина в неисправности.",
    "return": "Возврат: клиент хочет вернуть или обменять ИСПРАВНЫЙ товар (не подошёл, передумал, "
              "не тот цвет), спрашивает порядок возврата или когда вернут деньги за уже сданный товар.",
    "tech": "Техвопрос: сайт, приложение, личный кабинет — не могу войти, не приходит код, ошибка при "
            "оформлении; как настроить или подключить ИСПРАВНЫЙ товар, совместимость, где инструкция.",
    "spam": "Спам: не клиентское обращение — реклама услуг (SEO, лиды, кредиты), рассылка, фишинг "
            "со ссылкой, бессмыслица, сообщение явно не по адресу.",
}
PRIORITIES = ("high", "normal")

PRIORITY_POLICY = """Приоритет high, если выполнено хотя бы одно:
  - в тексте явная срочность: жёсткий срок, серьёзные последствия, крупная сумма, угроза претензии;
  - сегмент клиента key И тема одна из: payment, delivery, defect, tech.
Во всех остальных случаях normal. Спам всегда normal, даже если в нём написано «срочно»."""

SYSTEM = ("Ты система триажа обращений в поддержку интернет-магазина бытовой техники. "
          "Ты классифицируешь обращения и назначаешь ответственного строго по инструкции. "
          "Отвечаешь только JSON, без пояснений и markdown.")

# Всё неизменное стоит В НАЧАЛЕ, обращение — в конце: общий префикс попадает в prefix cache vLLM.
USER_TEMPLATE = """Категории тем:
{topics}

{priority_policy}

Справочник сотрудников (назначай по теме и сегменту):
{staff}

Если обращение упоминает несколько проблем, тему выбирай по главной просьбе клиента.

Ответ строго JSON без markdown:
{{"topic": "...", "priority": "high|normal", "assignee": "код"}}

Обращение:
[сегмент {segment}] {text}"""

DRAFT_SYSTEM = ("Ты оператор поддержки интернет-магазина. Пишешь черновик ответа клиенту, "
                "который потом прочитает и отправит человек. Вежливо, по делу, без выдуманных фактов: "
                "если нужны данные заказа, которых нет, так и напиши, что их нужно уточнить.")

TRIAGE_MAX_TOKENS = 64
DRAFT_MAX_TOKENS = 400


def load_tickets() -> list[dict]:
    return [json.loads(l) for l in open(DATA / "tickets.jsonl", encoding="utf-8") if l.strip()]


def load_directory() -> dict:
    return json.loads((DATA / "employees.json").read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def staff_text(d: dict) -> str:
    lines = []
    for e in d["employees"]:
        seg = f", только сегмент {'/'.join(e['segments'])}" if e.get("segments") else ""
        lines.append(f"  {e['code']} — {e['name']}, темы: {', '.join(e['topics'])}{seg}")
    lines.append(f"  {d['head']['code']} — {d['head']['name']}, эскалация при перегрузке")
    lines.append(f"  {d['spam_queue']} — очередь спама, не человек")
    return "\n".join(lines)


def candidates(topic: str, segment: str, d: dict) -> list[str]:
    """Кто вправе взять обращение: зависит только от темы и сегмента."""
    if topic == "spam":
        return [d["spam_queue"]]
    return [e["code"] for e in d["employees"]
            if topic in e["topics"] and (not e.get("segments") or segment in e["segments"])]


def triage_messages(ticket: dict, d: dict) -> list[dict]:
    user = USER_TEMPLATE.format(
        topics="\n".join(f"  {k} — {v}" for k, v in TOPICS.items()),
        priority_policy=PRIORITY_POLICY, staff=staff_text(d),
        segment=ticket.get("segment", "?"), text=normalize(ticket["text"]))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def draft_messages(ticket: dict, topic: str | None = None, priority: str | None = None) -> list[dict]:
    user = (f"Обращение клиента (тема {topic or ticket['topic']}, приоритет {priority or ticket['priority']}, "
            f"сегмент {ticket.get('segment', '?')}):\n\n{ticket['text']}\n\n"
            "Напиши черновик ответа, 3–6 предложений.")
    return [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}]


def parse_triage(content: str, d: dict) -> dict | None:
    """Невалидный ответ — None: в метриках это ошибка, а не пропуск."""
    s = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S)
    s = re.sub(r"```(?:json)?", "", s)
    try:
        item = json.loads(s[s.find("{"): s.rfind("}") + 1])
        topic = str(item["topic"]).strip().lower()
        priority = str(item["priority"]).strip().lower()
    except Exception:  # noqa: BLE001
        return None
    if topic not in TOPICS or priority not in PRIORITIES:
        return None
    codes = {e["code"] for e in d["employees"]} | {d["head"]["code"], d["spam_queue"]}
    assignee = str(item.get("assignee", "")).strip().lower()
    return {"topic": topic, "priority": priority, "assignee": assignee if assignee in codes else ""}


def build_requests(tickets: list[dict], d: dict) -> list[dict]:
    """Поток запросов для нагрузочного теста: по два на обращение (на спам — один)."""
    reqs = []
    for t in tickets:
        reqs.append({"kind": "triage", "id": t["id"], "messages": triage_messages(t, d),
                     "max_tokens": TRIAGE_MAX_TOKENS, "temperature": 0.0})
        if t["topic"] != "spam":
            reqs.append({"kind": "draft", "id": t["id"], "messages": draft_messages(t),
                         "max_tokens": DRAFT_MAX_TOKENS, "temperature": 0.7})
    return reqs
