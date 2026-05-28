"""
Chapter 6 — Three blackboard agents for the blog-post pipeline.

run_researcher: ReAct loop with web_search + read_url + submit_research.
                Tracks which URLs were actually opened; rejects submit_research
                if any claim URL was not read via read_url. This is the Ch 5
                retrofit — without it, agents satisfy the handoff schema with
                URLs they only saw in search snippets.

run_drafter:    Single LLM call. Takes research.claims, produces title + body
                + citations_used. Coordinator validates body word count and
                that every cited URL appears in body text.

run_editor:     Single LLM call. Takes draft, returns edited title + body
                + edits_made (non-empty list of changes).
"""
import json
import ollama

from agent import web_search, read_url
from state_store import now_iso

MODEL = "qwen3:14b"


# --- Researcher ---------------------------------------------------------------

RESEARCHER_SYSTEM = """You are a research agent. Given a topic, find at least 3 specific claims about it, each anchored to a URL you have actually READ via read_url (not just seen in a search snippet).

Process:
1. web_search to find candidate sources.
2. read_url on the most promising sources to see full content (NOT just snippets).
3. After reading >=3 distinct sources, call submit_research with a list of claims. Each claim:
   - Is a specific, verifiable statement (not a generic summary like "the paper is about transformers")
   - Has a 'url' field that is one of the URLs you OPENED via read_url

RULES:
- Use /no_think.
- One tool call per step.
- Do NOT cite a URL you have only seen in a search result — you must read_url it first.
- If a read_url returns no useful content, search again with a refined query.
"""

RESEARCHER_SCHEMA = {
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
                        "tool": {"const": "submit_research"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "claims": {
                                    "type": "array",
                                    "minItems": 3,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "url": {"type": "string"},
                                        },
                                        "required": ["text", "url"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["claims"],
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


def run_researcher(state: dict, max_steps: int = 15, verbose: bool = True) -> dict:
    topic = state["topic"]
    opened_urls: set[str] = set()
    metrics = {"steps": 0, "prompt_tokens": 0, "completion_tokens": 0, "tool_errors": 0}

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": f"/no_think\nTopic: {topic}\n\nDo the task."},
    ]

    for step in range(1, max_steps + 1):
        metrics["steps"] = step
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=messages,
                format=RESEARCHER_SCHEMA,
                options={"temperature": 0.2, "num_ctx": 8192},
                think=False,
            )
        except Exception as e:
            metrics["tool_errors"] += 1
            return {"research": None, "metrics": metrics, "error": f"ollama: {e}"}

        metrics["prompt_tokens"] += resp.get("prompt_eval_count", 0) or 0
        metrics["completion_tokens"] += resp.get("eval_count", 0) or 0

        content = resp["message"]["content"]
        try:
            parsed = json.loads(content)
            thought = parsed["thought"]
            action = parsed["action"]
            tool = action["tool"]
            args = action.get("args", {})
        except Exception as e:
            metrics["tool_errors"] += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Parse error: {e}. Emit valid action JSON."})
            continue

        if verbose:
            print(f"    [researcher step {step}] {tool}({json.dumps(args)[:120]})")

        if tool == "submit_research":
            claims = args["claims"]
            unread = [c["url"] for c in claims if c["url"] not in opened_urls]
            if unread:
                feedback = (
                    f"submit_research REJECTED. These claim URLs were never read via read_url: "
                    f"{unread}. You must call read_url(url) on every URL you intend to cite. "
                    f"URLs you have actually read: {sorted(opened_urls) or '[]'}. "
                    f"Either read those URLs first, or replace those claims with ones grounded in URLs you have read."
                )
                if verbose:
                    print(f"    [researcher] submit rejected: {len(unread)} unread URLs")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation:\n{feedback}"})
                continue

            research = {
                "claims": claims,
                "opened_urls": sorted(opened_urls),
                "completed_at": now_iso(),
            }
            return {"research": research, "metrics": metrics}

        try:
            if tool == "web_search":
                observation = web_search(args["query"])
            elif tool == "read_url":
                url = args["url"]
                observation = read_url(url)
                opened_urls.add(url)
            else:
                observation = f"<unknown tool: {tool}>"
        except Exception as e:
            metrics["tool_errors"] += 1
            observation = f"<tool error: {e}>"

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation:\n{observation[:2000]}"})

    return {"research": None, "metrics": metrics, "error": "max steps reached without submit"}


# --- Drafter -----------------------------------------------------------------

DRAFTER_SYSTEM = """You are a technical blog writer. Given research claims with source URLs, write a 500-700 word blog post.

REQUIREMENTS:
- Output JSON with: title, body, citations_used
- body must be 500-700 words
- citations_used: list at least 3 URLs from the research claims
- EVERY URL in citations_used MUST appear verbatim in the body text (as a bare URL or in a markdown link)
- Do NOT invent claims or cite URLs not in the research
- Use /no_think
"""

DRAFTER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "citations_used": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
    },
    "required": ["title", "body", "citations_used"],
    "additionalProperties": False,
}


def run_drafter(state: dict, retry_feedback: str | None = None, verbose: bool = True) -> dict:
    research = state["research"]
    metrics = {"steps": 1, "prompt_tokens": 0, "completion_tokens": 0, "tool_errors": 0}

    claims_block = "\n".join(
        f"- {c['text']}\n  url: {c['url']}" for c in research["claims"]
    )
    user_msg = (
        f"/no_think\nTopic: {state['topic']}\n\n"
        f"Research claims you must work from:\n{claims_block}\n\n"
        f"Write a 500-700 word blog post. Cite by embedding the URL in the body text. "
        f"List the URLs you cited in citations_used."
    )
    if retry_feedback:
        user_msg += f"\n\nPrevious attempt rejected: {retry_feedback}\nFix the issue."

    messages = [
        {"role": "system", "content": DRAFTER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            format=DRAFTER_SCHEMA,
            options={"temperature": 0.3, "num_ctx": 8192},
            think=False,
        )
    except Exception as e:
        return {"draft": None, "metrics": metrics, "error": f"ollama: {e}"}

    metrics["prompt_tokens"] = resp.get("prompt_eval_count", 0) or 0
    metrics["completion_tokens"] = resp.get("eval_count", 0) or 0

    if verbose:
        print(f"    [drafter] tokens prompt={metrics['prompt_tokens']} compl={metrics['completion_tokens']}")

    try:
        parsed = json.loads(resp["message"]["content"])
    except Exception as e:
        return {"draft": None, "metrics": metrics, "error": f"parse: {e}"}

    draft = {
        "title": parsed["title"],
        "body": parsed["body"],
        "citations_used": parsed["citations_used"],
        "completed_at": now_iso(),
    }
    return {"draft": draft, "metrics": metrics}


# --- Editor ------------------------------------------------------------------

EDITOR_SYSTEM = """You are a technical editor. Given a draft blog post, return an edited version.

REQUIREMENTS:
- Improve clarity, fix factual ambiguity, tighten prose
- PRESERVE every URL that appears in the body — do not delete or alter citations
- Maintain word count >=500
- edits_made: a non-empty list of specific changes you made (e.g., "Clarified the claim about X", "Removed redundant paragraph about Y", "Fixed citation spacing")
- Output JSON with: title, body, edits_made
- Use /no_think
"""

EDITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "edits_made": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
    },
    "required": ["title", "body", "edits_made"],
    "additionalProperties": False,
}


def run_editor(state: dict, retry_feedback: str | None = None, verbose: bool = True) -> dict:
    draft = state["draft"]
    metrics = {"steps": 1, "prompt_tokens": 0, "completion_tokens": 0, "tool_errors": 0}

    user_msg = (
        f"/no_think\nDraft to edit:\n\n"
        f"TITLE: {draft['title']}\n\nBODY:\n{draft['body']}\n\n"
        f"Edit this for clarity, accuracy, and tightness. Preserve every URL in the body. "
        f"Report your edits explicitly."
    )
    if retry_feedback:
        user_msg += f"\n\nPrevious attempt rejected: {retry_feedback}\nFix the issue."

    messages = [
        {"role": "system", "content": EDITOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            format=EDITOR_SCHEMA,
            options={"temperature": 0.3, "num_ctx": 8192},
            think=False,
        )
    except Exception as e:
        return {"final": None, "metrics": metrics, "error": f"ollama: {e}"}

    metrics["prompt_tokens"] = resp.get("prompt_eval_count", 0) or 0
    metrics["completion_tokens"] = resp.get("eval_count", 0) or 0

    if verbose:
        print(f"    [editor] tokens prompt={metrics['prompt_tokens']} compl={metrics['completion_tokens']}")

    try:
        parsed = json.loads(resp["message"]["content"])
    except Exception as e:
        return {"final": None, "metrics": metrics, "error": f"parse: {e}"}

    final = {
        "title": parsed["title"],
        "body": parsed["body"],
        "edits_made": parsed["edits_made"],
        "completed_at": now_iso(),
    }
    return {"final": final, "metrics": metrics}
