"""Копирует из Битрикс-кейса то, что нужно этому проекту: 150 размеченных обращений
и справочник сотрудников. После этого проект самодостаточен.

    uv run python scripts/prepare_data.py --src ../bitrix_kit/bitrix_kit
"""
import argparse
import json
from pathlib import Path

KEEP = ("id", "text", "segment", "topic", "priority", "source", "trap")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../bitrix_kit/bitrix_kit")
args = ap.parse_args()
src, dst = Path(args.src), Path(__file__).resolve().parent.parent / "data"
dst.mkdir(exist_ok=True)

rows = [json.loads(l) for l in open(src / "data/prospective.jsonl", encoding="utf-8") if l.strip()]
with open(dst / "tickets.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({k: r.get(k) for k in KEEP}, ensure_ascii=False) + "\n")
emp = json.loads((src / "employees.json").read_text(encoding="utf-8"))
emp.pop("_comment", None)
for e in emp["employees"]:
    e.pop("mock_user_id", None)
emp["head"].pop("mock_user_id", None)
(dst / "employees.json").write_text(json.dumps(emp, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"tickets: {len(rows)} -> {dst / 'tickets.jsonl'}")
