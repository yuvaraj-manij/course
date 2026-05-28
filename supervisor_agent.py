"""
Chapter 5 — Supervisor pattern for competitive intelligence briefs.

Supervisor routes to 4 specialists (products, news, competitors, hiring), each running
its own bounded ReAct loop with web_search + read_url. The supervisor never sees
individual search results — only section summaries. That isolation IS the pattern.
"""
import json
import time
import ollama

from agent import web_search, read_url

MODEL = "qwen3:14b"

# ============================================================
# Specialist (one code path, parameterized by role)
# ============================================================

SPECIALIST_PROMPTS = {
    "products": """You are the PRODUCTS specialist for a competitive intelligence brief on {company}.
Gather information about {company}'s main products and services.

Use web_search to find product pages, blog posts, or recent reviews. Use read_url on at most 2 promising results.
After 1-2 searches and 1-2 URL reads, call return_section with a 100-150 word summary of their main products,
target customers, and any notable product strategy. Cite the URLs you actually read.

Focus ONLY on products. Do not write about news, competitors, or hiring.""",

    "news": """You are the NEWS specialist for a competitive intelligence brief on {company}.
Gather recent news (funding, leadership, product launches, partnerships, controversies).

Use web_search with queries like '{company} news 2026' or '{company} announcement'.
After 1-2 searches and 1-2 URL reads, call return_section with a 100-150 word summary of the most notable recent events.
Cite the URLs you actually read.

Focus ONLY on news. Do not write about products, competitors, or hiring.""",

    "competitors": """You are the COMPETITORS specialist for a competitive intelligence brief on {company}.
Identify {company}'s 2-4 key competitors and how it's positioned against them.

Use web_search with queries like '{company} vs' or '{company} competitors'.
After 1-2 searches and 1-2 URL reads, call return_section with a 100-150 word summary naming the competitors
and the dimensions on which they compete (pricing, features, market segment).
Cite the URLs you actually read.

Focus ONLY on competitors. Do not write about products, news, or hiring.""",

    "hiring": """You are the HIRING specialist for a competitive intelligence brief on {company}.
Identify what roles {company} is hiring for and what that signals about their strategy.

Use web_search with queries like '{company} careers' or '{company} hiring engineering 2026'.
After 1-2 searches and 1-2 URL reads, call return_section with a 100-150 word summary of hiring focus areas
and any signal about company priorities (research-heavy, sales-heavy, infra build-out, etc.).
Cite the URLs you actually read.

Focus ONLY on hiring. Do not write about products, news, or competitors.""",
}

SPECIALIST_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "web_search"},
                        "args": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "read_url"},
                        "args": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "return_section"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["text", "sources"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
            ],
        },
    },
    "required": ["thought", "action"],
    "additionalProperties": False,
}


def _specialist_dispatch(tool: str, args: dict):
    if tool == "web_search":
        return web_search(args["query"])
    if tool == "read_url":
        return read_url(args["url"])
    return f"<unknown tool: {tool}>"


def run_specialist(company: str, role: str, max_steps: int = 8, verbose: bool = True) -> dict:
    metrics = {
        "role": role,
        "wall_clock_s": 0.0,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "completed": False,
        "section_text": None,
        "sources": [],
    }
    start = time.time()
    messages = [
        {"role": "system", "content": SPECIALIST_PROMPTS[role].format(company=company)},
        {"role": "user", "content": f"/no_think\nResearch the {role} section for {company} and return it."},
    ]

    for step in range(1, max_steps + 1):
        metrics["steps"] = step
        try:
            resp = ollama.chat(
                model=MODEL, messages=messages, format=SPECIALIST_SCHEMA,
                options={"temperature": 0.2, "num_ctx": 8192}, think=False,
            )
        except Exception as e:
            print(f"      [{role} ollama error: {e}]", flush=True)
            break

        metrics["prompt_tokens"] += resp.get("prompt_eval_count", 0) or 0
        metrics["completion_tokens"] += resp.get("eval_count", 0) or 0
        content = resp["message"]["content"]

        try:
            parsed = json.loads(content)
            tool = parsed["action"]["tool"]
            args = parsed["action"].get("args", {})
        except (json.JSONDecodeError, KeyError):
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Emit valid action JSON."})
            continue

        if verbose:
            arg_preview = json.dumps(args)[:120]
            print(f"      [{role} step {step}] {tool}({arg_preview})", flush=True)

        if tool == "return_section":
            metrics["completed"] = True
            metrics["section_text"] = args["text"]
            metrics["sources"] = args["sources"]
            break

        try:
            observation = _specialist_dispatch(tool, args)
        except Exception as e:
            observation = f"<tool error: {e}>"

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation:\n{observation[:2000]}"})

    metrics["wall_clock_s"] = round(time.time() - start, 2)
    if not metrics["completed"]:
        metrics["section_text"] = f"[{role} specialist did not complete in {max_steps} steps]"
    return metrics


# ============================================================
# Supervisor
# ============================================================

SUPERVISOR_PROMPT = """You are the SUPERVISOR producing a one-page competitive intelligence brief on {company}.

You have four specialists. Each runs its own research; you only coordinate.
- dispatch_products: calls the Products specialist
- dispatch_news: calls the News specialist
- dispatch_competitors: calls the Competitors specialist
- dispatch_hiring: calls the Hiring specialist

Workflow:
1. Call each specialist exactly once (any order).
2. After all four return, call final_brief(text, sources) with a synthesized brief.

The final brief should have 4 labelled sections (Products / News / Competitors / Hiring) and
end with a 1-sentence overall assessment. Cite all URLs the specialists returned, deduplicated.

You do NOT do web research yourself. Only the specialists do that.

RULES:
- /no_think
- Each specialist called exactly once. No duplicates.
- Cite only URLs returned by specialists. No fabrication.
"""

# Build the supervisor schema — 4 dispatch tools + final_brief
_DISPATCH_TOOLS = [
    {
        "type": "object",
        "properties": {
            "tool": {"const": f"dispatch_{role}"},
            "args": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }
    for role in ("products", "news", "competitors", "hiring")
]

SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "object",
            "oneOf": _DISPATCH_TOOLS + [
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "final_brief"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["text", "sources"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
            ],
        },
    },
    "required": ["thought", "action"],
    "additionalProperties": False,
}


def run_supervisor(company: str, max_steps: int = 10, verbose: bool = True) -> dict:
    metrics = {
        "company": company,
        "wall_clock_s": 0.0,
        "supervisor_steps": 0,
        "supervisor_prompt_tokens": 0,
        "supervisor_completion_tokens": 0,
        "specialists": {},
        "completed": False,
        "final_text": None,
        "sources": [],
    }
    start = time.time()
    dispatched: set = set()

    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT.format(company=company)},
        {"role": "user", "content": f"/no_think\nProduce the competitive intelligence brief for {company}."},
    ]

    for step in range(1, max_steps + 1):
        metrics["supervisor_steps"] = step
        try:
            resp = ollama.chat(
                model=MODEL, messages=messages, format=SUPERVISOR_SCHEMA,
                options={"temperature": 0.2, "num_ctx": 8192}, think=False,
            )
        except Exception as e:
            print(f"  [supervisor ollama error: {e}]", flush=True)
            break

        metrics["supervisor_prompt_tokens"] += resp.get("prompt_eval_count", 0) or 0
        metrics["supervisor_completion_tokens"] += resp.get("eval_count", 0) or 0
        content = resp["message"]["content"]

        try:
            parsed = json.loads(content)
            thought = parsed["thought"]
            tool = parsed["action"]["tool"]
            args = parsed["action"].get("args", {})
        except (json.JSONDecodeError, KeyError):
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Emit valid action JSON."})
            continue

        if verbose:
            print(f"\n[Supervisor step {step}]")
            print(f"  Thought: {thought[:200]}")
            print(f"  Action:  {tool}")

        if tool == "final_brief":
            metrics["completed"] = True
            metrics["final_text"] = args["text"]
            metrics["sources"] = args["sources"]
            break

        if tool.startswith("dispatch_"):
            role = tool.removeprefix("dispatch_")
            if role in dispatched:
                obs = f"You already called dispatch_{role}. Move to a different specialist or final_brief."
            else:
                print(f"  → Dispatching {role} specialist...", flush=True)
                spec = run_specialist(company, role, verbose=verbose)
                metrics["specialists"][role] = spec
                dispatched.add(role)
                src_str = "\n".join(f"- {s}" for s in spec["sources"]) or "<no sources>"
                obs = (f"[{role} specialist returned: steps={spec['steps']}, "
                       f"time={spec['wall_clock_s']}s, completed={spec['completed']}]\n\n"
                       f"SECTION:\n{spec['section_text']}\n\n"
                       f"SOURCES:\n{src_str}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Unknown tool: {tool}"})

    metrics["wall_clock_s"] = round(time.time() - start, 2)
    metrics["total_prompt_tokens"] = (
        metrics["supervisor_prompt_tokens"]
        + sum(s["prompt_tokens"] for s in metrics["specialists"].values())
    )
    metrics["total_completion_tokens"] = (
        metrics["supervisor_completion_tokens"]
        + sum(s["completion_tokens"] for s in metrics["specialists"].values())
    )
    metrics["total_steps"] = metrics["supervisor_steps"] + sum(
        s["steps"] for s in metrics["specialists"].values()
    )
    return metrics


if __name__ == "__main__":
    import sys
    company = sys.argv[1] if len(sys.argv) > 1 else "Anthropic"
    m = run_supervisor(company, max_steps=10, verbose=True)
    print("\n" + "=" * 60)
    print(f"Company:            {m['company']}")
    print(f"Completed:          {m['completed']}")
    print(f"Total wall clock:   {m['wall_clock_s']}s")
    print(f"Supervisor steps:   {m['supervisor_steps']}")
    print(f"Total agent steps:  {m['total_steps']}")
    print(f"Total prompt tok:   {m['total_prompt_tokens']}")
    print(f"Total compl tok:    {m['total_completion_tokens']}")
    print("Per specialist:")
    for role, s in m["specialists"].items():
        print(f"  {role:12s}  steps={s['steps']:2d}  time={s['wall_clock_s']:6.1f}s  "
              f"tok={s['prompt_tokens']+s['completion_tokens']:6d}  done={s['completed']}")
    print(f"Sources:            {len(m['sources'])} URLs")
    print("=" * 60)
    if m["final_text"]:
        print("\nFINAL BRIEF:")
        print(m["final_text"])
