"""
Chapter 4 — Single-agent version of the paper-summary task.

One ReAct loop, schema-constrained, three tools (search_arxiv, get_paper, final_answer).
Captures metrics for the head-to-head benchmark in bench.py.
"""
import json
import time
import ollama

from paper_tools import search_papers, get_paper

MODEL = "qwen3:14b"

SYSTEM_PROMPT = """You are a research assistant. Given a paper title, you must:

1. Find the paper via search_papers (if the first 3 results miss it, try a more specific query or raise max_results to 10).
2. Read its full abstract via get_paper(paper_id).
3. Identify 3 key claims the paper makes (be specific — not "the paper is about transformers").
4. For each claim, search for ONE supporting paper and ONE contradicting paper.
   - You judge supporting vs contradicting based on the abstract you read for each candidate.
   - If you genuinely cannot find a contradicting paper for a claim, say "no clear contradiction found" in your final answer rather than fabricating one.
5. Produce a 500-word summary via final_answer that:
   - States the main paper's contribution
   - Lists the 3 key claims
   - For each claim, names the supporting and contradicting paper (with paper_id) and briefly explains why
   - Cites only abs_url values you have actually seen in search_papers results or retrieved with get_paper

TOOLS:
- search_papers(query: str, max_results: int = 5) — search OpenAlex, returns list of {paper_id, title, authors, published, abstract_preview, abs_url, cited_by_count}
- get_paper(paper_id: str) — fetch full abstract for one paper (paper_id looks like 'W2741809807')
- final_answer(text: str, sources: list[str]) — emit your final summary and the list of abs_urls you cited

RULES:
- Use /no_think — no <think> blocks.
- One tool call per step.
- Cite only URLs you have actually seen. No fabrication.
- If a search returns nothing useful, refine the query and try again — don't invent a paper.
"""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "search_papers"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                            },
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
                        "tool": {"const": "get_paper"},
                        "args": {
                            "type": "object",
                            "properties": {"paper_id": {"type": "string"}},
                            "required": ["paper_id"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "final_answer"},
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


def _dispatch(tool: str, args: dict):
    if tool == "search_papers":
        results = search_papers(args["query"], args.get("max_results", 5))
        lines = [f"{i+1}. [{r['paper_id']}] {r['title']} (cited {r['cited_by_count']}x)\n   {r['abstract_preview']}"
                 for i, r in enumerate(results)]
        return "\n".join(lines) if lines else "<no results>"
    if tool == "get_paper":
        p = get_paper(args["paper_id"])
        if "error" in p:
            return p["error"]
        return (f"title: {p['title']}\nauthors: {', '.join(p['authors'][:3])}\n"
                f"published: {p['published']}\nabs_url: {p['abs_url']}\n\nabstract: {p['abstract']}")
    return f"<unknown tool: {tool}>"


def run_agent(paper_title: str, max_steps: int = 25, verbose: bool = True) -> dict:
    metrics = {
        "paper_title": paper_title,
        "wall_clock_s": 0.0,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_errors": 0,
        "completed": False,
        "final_text": None,
        "sources": [],
        "trace": [],
    }
    start = time.time()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"/no_think\nPaper title: {paper_title}\n\nDo the task."},
    ]

    for step in range(1, max_steps + 1):
        metrics["steps"] = step
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=messages,
                format=ACTION_SCHEMA,
                options={"temperature": 0.2, "num_ctx": 8192},
                think=False,
            )
        except Exception as e:
            metrics["tool_errors"] += 1
            metrics["trace"].append({"step": step, "error": f"ollama: {e}"})
            break

        metrics["prompt_tokens"] += resp.get("prompt_eval_count", 0) or 0
        metrics["completion_tokens"] += resp.get("eval_count", 0) or 0

        content = resp["message"]["content"]
        try:
            parsed = json.loads(content)
            thought = parsed["thought"]
            action = parsed["action"]
            tool = action["tool"]
            args = action.get("args", {})
        except (json.JSONDecodeError, KeyError) as e:
            metrics["tool_errors"] += 1
            metrics["trace"].append({"step": step, "parse_error": str(e), "raw": content[:200]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Parse error: {e}. Emit valid action JSON."})
            continue

        if verbose:
            print(f"\n── Step {step} ──────────────────────")
            print(f"Thought: {thought[:200]}")
            print(f"Action: {tool}({json.dumps(args)[:200]})")

        if tool == "final_answer":
            metrics["completed"] = True
            metrics["final_text"] = args["text"]
            metrics["sources"] = args["sources"]
            metrics["trace"].append({"step": step, "thought": thought, "action": "final_answer"})
            if verbose:
                print(f"\nFINAL ANSWER ({len(args['text'])} chars, {len(args['sources'])} sources)")
            break

        try:
            observation = _dispatch(tool, args)
        except Exception as e:
            metrics["tool_errors"] += 1
            observation = f"<tool error: {e}>"

        if verbose:
            preview = observation[:300].replace("\n", " | ")
            print(f"Observation: {preview}{'...' if len(observation) > 300 else ''}")

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})
        metrics["trace"].append({"step": step, "thought": thought, "tool": tool, "args": args})

    metrics["wall_clock_s"] = round(time.time() - start, 2)
    return metrics


if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else "Attention Is All You Need"
    m = run_agent(title, max_steps=25, verbose=True)
    print("\n" + "=" * 60)
    print(f"completed:         {m['completed']}")
    print(f"steps:             {m['steps']}")
    print(f"wall_clock_s:      {m['wall_clock_s']}")
    print(f"prompt_tokens:     {m['prompt_tokens']}")
    print(f"completion_tokens: {m['completion_tokens']}")
    print(f"tool_errors:       {m['tool_errors']}")
    print(f"sources:           {m['sources']}")
    print("=" * 60)
    if m["final_text"]:
        print("\nFINAL TEXT:")
        print(m["final_text"])
