"""Вариант A: квантует Qwen2.5-7B-Instruct в FP8 (W8A8) через llm-compressor.

Схема FP8_DYNAMIC: веса — статически, по каналам; активации — динамически на инференсе.
Калибровочные данные не нужны, поэтому квантование идёт на CPU за минуты и не требует
держать FP16-модель (15 ГБ) в видеопамяти, куда она не помещается.
`lm_head` не квантуется — это стандартная рекомендация: выходной слой чувствительнее остальных.

Запускается внутри контейнера vLLM (там уже есть torch и transformers):

    docker run --rm --entrypoint bash -v own-llm-vs-api_hf-cache:/root/.cache/huggingface \
      -v own-llm-vs-api_models:/models -v ./scripts:/scripts vllm/vllm-openai:v0.33.0 \
      -c "pip install -q llmcompressor && python3 /scripts/quantize_fp8.py"
"""
import time

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen2.5-7B-Instruct"
DST = "/models/Qwen2.5-7B-Instruct-FP8-dynamic"

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(SRC)
print(f"загружено за {time.time() - t0:.0f} с")

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)

model.save_pretrained(DST)
tokenizer.save_pretrained(DST)
print(f"готово за {time.time() - t0:.0f} с -> {DST}")
