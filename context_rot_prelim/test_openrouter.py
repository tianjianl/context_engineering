#!/usr/bin/env python3
"""Quick smoke test for OpenRouter API endpoints.

Usage:
    source ~/.bashrc
    python -m context_rot_prelim.test_openrouter
"""

import os
import sys

from openai import OpenAI

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("OPENROUTER_API_KEY not set. Run: source ~/.bashrc")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

MODELS = [
    "minimax/minimax-m2.5",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-5",
    "deepseek/deepseek-v3.2",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-397b-a17b",
]

PROMPT = "What is 2 + 3? Answer with just the number."

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=128,
            temperature=0.0,
        )
        choice = resp.choices[0]
        print(f"  Response: {choice.message.content.strip()[:200]}")
        print(f"  Finish:   {choice.finish_reason}")
        if resp.usage:
            print(f"  Tokens:   prompt={resp.usage.prompt_tokens}, "
                  f"completion={resp.usage.completion_tokens}")
        print("  Status:   OK")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print(f"\n{'='*60}")
print("Done.")
