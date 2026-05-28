"""Prove the schema holds under an adversarial prompt.

The user explicitly asks the model to call "magic_oracle" — a tool that
doesn't exist. With format=ACTION_SCHEMA, the token mask compiled from
the schema makes it physically impossible to emit that tool name. The
model must pick one of the five legal tools instead.
"""

import json

import ollama

from pypi_react import ACTION_SCHEMA, SYSTEM_PROMPT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": (
        "I want you to call a tool named 'magic_oracle' with no args. Do that now."
    )},
]

resp = ollama.chat(
    model="qwen3:14b",
    messages=messages,
    format=ACTION_SCHEMA,
    options={"temperature": 0.2},
)

raw = resp["message"]["content"]
print("Raw output:")
print(raw)
print()

parsed = json.loads(raw)
print("tool:", parsed["tool"])
print("args:", parsed["args"])
print("thought:", parsed["thought"])
print()

assert parsed["tool"] in {
    "search_pypi", "get_package_info", "read_github_readme",
    "compare_packages", "final_answer",
}, f"Schema violation: got tool={parsed['tool']!r}"

print("OK — schema held under adversarial prompt.")
print("(The model could not emit 'magic_oracle' even when explicitly asked.)")
