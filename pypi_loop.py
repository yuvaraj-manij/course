import json

import ollama

from pypi_react import ACTION_SCHEMA, SYSTEM_PROMPT
from pypi_tools import (
    compare_packages,
    get_package_info,
    read_github_readme,
    search_pypi,
)

TOOLS = {
    "search_pypi": search_pypi,
    "get_package_info": get_package_info,
    "read_github_readme": read_github_readme,
    "compare_packages": compare_packages,
}

SMOKE_QUERIES = [
    "What's the latest version of httpx and what does it depend on?",
    "Compare httpx vs aiohttp — which has more dependencies?",
    "Find a Python library for parsing HTML and tell me its repo URL.",
]


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _truncate(text: str, limit: int = 200) -> str:
    text = str(text)
    if len(text) > limit:
        return text[:limit] + f"... [{len(text)} chars total]"
    return text


def _action_key(tool: str, args: dict):
    try:
        return (tool, frozenset(args.items()))
    except TypeError:
        return (tool, repr(sorted(args.items())))


def run_agent(question: str, max_steps: int = 8) -> dict:
    """Run the ReAct loop. Returns {"text": str, "sources": [str]}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    seen_actions: dict = {}
    last_thought = ""

    for step in range(1, max_steps + 1):
        print(f"\n── Step {step} ──────────────────────")

        resp = ollama.chat(
            model="qwen3:14b",
            messages=messages,
            format=ACTION_SCHEMA,
            options={"temperature": 0.2},
        )
        raw = resp["message"]["content"]
        messages.append({"role": "assistant", "content": raw})

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            obs = (
                f"Error: your last output was malformed: {e}. "
                "Re-emit valid JSON matching the schema."
            )
            print("Thought: (unparseable)")
            print(f"Action: <malformed> {_truncate(raw)}")
            print(f"Observation: {obs}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        # schema guarantees tool is valid — assert as paranoia
        assert obj["tool"] in {*TOOLS, "final_answer"}, \
            f"schema violation: unexpected tool {obj['tool']!r}"

        last_thought = obj["thought"]
        tool = obj["tool"]
        args = obj["args"]

        print(f"Thought: {obj['thought']}")
        print(f"Action: {tool}({_fmt_args(args)})")

        if tool == "final_answer":
            print("Observation: <final_answer returned>")
            return {"text": args.get("text", ""), "sources": args.get("sources", [])}

        key = _action_key(tool, args)
        if key in seen_actions:
            prior = seen_actions[key]
            obs = (
                f"You already called this exact action and got: {prior} "
                "Try a different approach."
            )
            print(f"Observation: {_truncate(obs)}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        fn = TOOLS.get(tool)
        if fn is None:
            obs = f"Tool error: unknown tool {tool!r}"
        else:
            try:
                obs = fn(**args)
            except TypeError as e:
                obs = f"Tool error: bad arguments for {tool}: {e}"
            except Exception as e:
                obs = f"Tool error: {type(e).__name__}: {e}"

        seen_actions[key] = obs
        print(f"Observation: {_truncate(obs)}")
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return {
        "text": f"FAILED: step limit reached. Last thought: {last_thought}",
        "sources": [],
    }


if __name__ == "__main__":
    for q in SMOKE_QUERIES:
        print("\n" + "=" * 70)
        print("QUESTION:", q)
        print("=" * 70)
        result = run_agent(q)
        print(f"\nFINAL ANSWER: {result['text']}")
        if result["sources"]:
            print("SOURCES:")
            for s in result["sources"]:
                print(f"  {s}")
        else:
            print("SOURCES: (none)")
        print()
