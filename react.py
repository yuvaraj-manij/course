import json

SYSTEM_PROMPT = """You are a ReAct agent that answers questions by using tools.

You have these tools:
- web_search(query: str) — Search the web via DuckDuckGo. Returns up to 5 results, each with title, URL, and snippet.
- read_url(url: str) — Fetch a URL and return the extracted main text (~3000 chars). Use this to read pages found via web_search.
- calculator(expr: str) — Evaluate a numeric expression. Supports +, -, *, /, **, %, parentheses, and unary minus. Numbers only — no variables or function calls.
- final_answer(text: str) — Call this when you have the answer. The agent loop stops after this.

On every turn, output a single JSON object with this exact schema:
{
  "thought": "<your reasoning for this step>",
  "tool": "<one of: web_search, read_url, calculator, final_answer>",
  "args": { ... arguments for that tool ... }
}

Rules:
- Output ONLY the JSON object. No markdown fences, no commentary before or after, no trailing text.
- Use calculator for ALL arithmetic, even things you think you can do in your head. Mental math is not allowed.
- Snippets from web_search are previews, not sources. Before giving a final_answer based on web information, read_url the most relevant result to confirm.
- Do not repeat an action you have already taken with identical args. If a tool returned a bad result, change the query, try a different URL, or rephrase — don't retry the same call.
- If a tool returns an error string or "No extractable content from <url>", treat it as information: pick a different URL or refine your query.
- In final_answer, include the URL(s) that support the answer so the user can verify.
- Cite ONLY URLs you have actually opened with read_url in this trajectory. Never cite a URL from a search snippet alone, and never cite a source you did not retrieve via a tool call. If you cannot read any sources, say so in the final_answer instead of citing.
- Keep "thought" short — one or two sentences explaining why this step, not a recap of everything so far.
"""

ALLOWED_TOOLS = {"web_search", "read_url", "calculator", "final_answer"}


class MalformedAction(Exception):
    pass


def parse_action(model_output: str) -> dict:
    try:
        obj = json.loads(model_output)
    except json.JSONDecodeError as e:
        raise MalformedAction(f"not valid JSON: {e}")

    if not isinstance(obj, dict):
        raise MalformedAction("top-level value must be a JSON object")

    for key in ("thought", "tool", "args"):
        if key not in obj:
            raise MalformedAction(f"missing required key: {key!r}")

    tool = obj["tool"]
    if tool not in ALLOWED_TOOLS:
        raise MalformedAction(f"unknown tool: {tool!r}")

    if not isinstance(obj["args"], dict):
        raise MalformedAction("'args' must be a JSON object")

    return {
        "thought": obj["thought"],
        "tool": tool,
        "args": obj["args"],
        "is_final": tool == "final_answer",
    }


if __name__ == "__main__":
    cases = [
        '{"thought": "search for it", "tool": "web_search", "args": {"query": "Eiffel Tower year"}}',
        '{"thought": "read the page", "tool": "read_url", "args": {"url": "https://example.com"}}',
        '{"thought": "compute", "tool": "calculator", "args": {"expr": "17**7 + 1889"}}',
        '{"thought": "done", "tool": "final_answer", "args": {"text": "1889"}}',
        '{"thought": "broken", "tool": "imaginary_tool", "args": {}}',
    ]
    for c in cases:
        try:
            print("OK:", parse_action(c))
        except MalformedAction as e:
            print("REJECTED:", e)
