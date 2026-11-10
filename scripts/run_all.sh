#!/usr/bin/env bash
# Полный прогон вариантов: запуск -> качество (поддержка) -> нагрузка -> качество (DocCheck).
#   bash scripts/run_all.sh B C D A E
set -u
export PYTHONIOENCODING=utf-8
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCCHECK="${DOCCHECK_DIR:-$ROOT/../DocCheck}"
CLAIMS="${DOCCHECK_CLAIMS:-eval-20261107-180552-analyst-only}"
cd "$ROOT"; mkdir -p results/raw

doccheck() {  # $1 = вариант, $2 = base url, $3 = модель
  [ -d "$DOCCHECK" ] || { echo "DocCheck не найден, пропуск"; return; }
  ( cd "$DOCCHECK" && DOCCHECK_ANALYST_BASE_URL="$2" DOCCHECK_ANALYST_MODEL="$3" DOCCHECK_MAX_OUTPUT_TOKENS=2000 \
      uv run python scripts/eval_planted.py --label "own-$1" --no-critic --reuse-claims "$CLAIMS" \
      > "$ROOT/results/raw/log_doccheck_$1.txt" 2>&1
    f=$(ls -t evals/eval-*-own-"$1".json 2>/dev/null | head -1)
    [ -n "$f" ] && cp "$f" "$ROOT/results/doccheck_$1.json"
    # чужой репозиторий оставляем как был
    rm -rf evals/eval-*-own-"$1".json evals/runs/eval-*-own-"$1"; git checkout -q evals/results.md )
}

for V in "$@"; do
  echo "=== $V $(date +%H:%M)"
  if [ "$V" = "E" ]; then TARGET=ollama; MODEL=qwen2.5:7b-instruct-q4_K_M; URL=http://127.0.0.1:11434/v1; OPEN="0.25,0.5,1,2,3,4,6,8"
  else TARGET=vllm; MODEL=local; URL=http://127.0.0.1:8000/v1; OPEN="1,2,4,6,8,10,12,14,16,18,20,24"; fi
  uv run python scripts/serve.py "$V" 2>&1 | tail -3 || { echo "$V: не поднялся"; continue; }
  curl -sf -m 5 "$URL/models" > /dev/null || { echo "$V: сервер недоступен"; continue; }
  uv run python scripts/eval_triage.py --target $TARGET --model "$MODEL" --name "$V" > "results/raw/log_quality_$V.txt" 2>&1
  uv run python scripts/loadtest.py --target $TARGET --model "$MODEL" --name "$V" --open "$OPEN" 2>&1 | tee "results/raw/log_load_$V.txt"
  doccheck "$V" "$URL" "$MODEL"
  echo "=== $V готов $(date +%H:%M)"
done
uv run python scripts/serve.py down > /dev/null 2>&1
