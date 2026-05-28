import json

SYSTEM_PROMPT = """You are a ReAct agent that answers questions about Python packages using PyPI tools.

Your output must be a JSON object describing your next action. The runtime enforces the structure — focus on choosing the right tool and args.

You have these tools:
- search_pypi(query) — search PyPI for packages matching a query; returns a numbered list of package names and descriptions.
- get_package_info(name) — fetch version, summary, release date, Python requirement, dependencies, and repo URL for a package from the PyPI JSON API.
- read_github_readme(repo_url) — fetch the raw README from a github.com repo URL; returns plain text up to ~12000 chars.
- compare_packages(name_a, name_b) — structured side-by-side comparison of two packages; preferred over calling get_package_info twice.
- final_answer(text, sources) — submit your answer. sources must be a list of URLs you actually retrieved in this trajectory.

Rules:
- When comparing two packages, call compare_packages once — do not call get_package_info twice manually.
- If get_package_info returns "not found", do not invent metadata for that package. Use search_pypi to find the correct package name, then retry.
- Cite only URLs you have actually fetched with read_github_readme, or canonical URLs confirmed by get_package_info (e.g. the repo_url field). Never cite a URL from a search snippet you did not open.
- sources in final_answer must list only URLs you retrieved in this trajectory. If you retrieved nothing, pass an empty list and say so in text.
- Do not repeat an action you have already taken with identical args. If a tool returned a bad result, try a different query or package name.
- If read_github_readme returns "No README found", try the repo URL with a trailing slash stripped, or note the absence in your answer.
- Keep "thought" short — one or two sentences on why this step, not a recap of everything so far.
"""

def _branch(tool_name: str, args_schema: dict) -> dict:
    """Build a oneOf branch with thought/tool/args all required.

    llama.cpp's grammar compiler processes each oneOf branch in isolation
    and does not inherit the parent object's required list. Putting thought
    in every branch is the only way to guarantee it appears in the output.
    """
    return {
        "properties": {
            "thought": {"type": "string"},
            "tool": {"const": tool_name},
            "args": args_schema,
        },
        "required": ["thought", "tool", "args"],
    }


ACTION_SCHEMA = {
    "type": "object",
    "oneOf": [
        _branch("search_pypi", {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        }),
        _branch("get_package_info", {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }),
        _branch("read_github_readme", {
            "type": "object",
            "required": ["repo_url"],
            "properties": {"repo_url": {"type": "string"}},
            "additionalProperties": False,
        }),
        _branch("compare_packages", {
            "type": "object",
            "required": ["name_a", "name_b"],
            "properties": {
                "name_a": {"type": "string"},
                "name_b": {"type": "string"},
            },
            "additionalProperties": False,
        }),
        _branch("final_answer", {
            "type": "object",
            "required": ["text", "sources"],
            "properties": {
                "text": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }),
    ],
}

ALLOWED_TOOLS = {
    "search_pypi", "get_package_info", "read_github_readme",
    "compare_packages", "final_answer",
}


class MalformedAction(Exception):
    pass


def parse_action(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise MalformedAction(f"not valid JSON: {e}")
    # Schema guarantees these, but keep as a hard assertion so a bug in
    # Ollama's structured-output path surfaces immediately rather than
    # causing a silent wrong dispatch.
    assert obj["tool"] in ALLOWED_TOOLS, f"unexpected tool from schema: {obj['tool']!r}"
    return {
        "thought": obj["thought"],
        "tool": obj["tool"],
        "args": obj["args"],
        "is_final": obj["tool"] == "final_answer",
    }
