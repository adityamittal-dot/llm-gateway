"""Day-1 check: the stock OpenAI SDK works against the gateway with only base_url changed.

Run the gateway (`uv run llm-gateway`) and Ollama, then:
    uv run python scripts/smoke.py [model]
"""

import sys
import time

from openai import OpenAI

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b-instruct-q8_0"
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
messages = [{"role": "user", "content": "Reply with one short sentence: what is a circuit breaker?"}]

print("models:", [m.id for m in client.models.list().data])

start = time.perf_counter()
resp = client.chat.completions.create(model=model, messages=messages, max_tokens=60, temperature=0)
print(f"non-stream ({time.perf_counter() - start:.1f}s):", resp.choices[0].message.content)
print("finish_reason:", resp.choices[0].finish_reason, "| usage:", resp.usage)

start = time.perf_counter()
first_token_at = None
parts = []
for chunk in client.chat.completions.create(
    model=model, messages=messages, max_tokens=60, temperature=0, stream=True
):
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        first_token_at = first_token_at or time.perf_counter()
        parts.append(delta)
print(f"stream (TTFT {first_token_at - start:.2f}s, {len(parts)} chunks):", "".join(parts))
