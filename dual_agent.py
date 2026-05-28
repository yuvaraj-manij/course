"""
Chapter 4 — Two-agent version of the paper-summary task.

Researcher (ReAct loop) gathers a structured packet → Writer (single call) synthesizes the summary.
The Writer has no tools. It cannot fetch more. Context isolation is the point.
"""
import json
import time
import ollama

from paper_tools import search_papers, get_paper

MODEL = "qwen3:14b"

# ============================================================
# Researcher
# ============================================================

RESEARCHER_PROMPT = """You are a research gatherer. Given a paper title, your job is to:

1. Find the paper via search_papers.
2. Read its full abstract via get_paper(paper_id).
3. Identify 3 specific claims the paper makes.
4. For each claim, find ONE supporting paper and ONE contradicting paper via search_papers + get_paper.
5. Submit the structured packet via submit_research.

You do NOT write the summary. The Writer agent does that. You only gather and structure the evidence.

If you genuinely cannot find a contradicting paper for a claim, set its paper_id to "" and title to "no clear contradiction found". Do not fabricate.

TOOLS:
- search_papers(query: str, max_results: int = 5)
- get_paper(paper_id: str)
- submit_research(main_paper, claims) — terminal action, emits the packet

RULES:
- /no_think
- One tool call per step.
- Cite only paper_ids you've actually seen in search results or retrieved.
- Each "why_supports" / "why_contradicts" should be one sentence grounded in the candidate's abstract.
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
                        "tool": {"const": "submit_research"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "main_paper": {
                                    "type": "object",
                                    "properties": {
                                        "paper_id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "abs_url": {"type": "string"},
                                        "abstract": {"type": "string"},
                                    },
                                    "required": ["paper_id", "title", "abs_url", "abstract"],
                                    "additionalProperties": False,
                                },
                                "claims": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "claim": {"type": "string"},
                                            "supporting": {
                                                "type": "object",
                                                "properties": {
                                                    "paper_id": {"type": "string"},
                                                    "title": {"type": "string"},
                                                    "abs_url": {"type": "string"},
                                                    "why_supports": {"type": "string"},
                                                },
                                                "required": ["paper_id", "title", "abs_url", "why_supports"],
                                                "additionalProperties": False,
                                            },
                                            "contradicting": {
                                                "type": "object",
                                                "properties": {
                                                    "paper_id": {"type": "string"},
                                                    "title": {"type": "string"},
                                                    "abs_url": {"type": "string"},
                                                    "why_contradicts": {"type": "string"},
                                                },
                                                "required": ["paper_id", "title", "abs_url", "why_contradicts"],
                                                "additionalProperties": False,
                                            },
                                        },
                                        "required": ["claim", "supporting", "contradicting"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["main_paper", "claims"],
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


def _validate_packet(packet: dict) -> str | None:
    """
    Semantic validator for submit_research. Schema enforces structure; this enforces meaning.
    Returns None if packet is acceptable, else a rejection message the Researcher sees as an observation.

    Rules:
    - Each claim must have a non-empty supporting paper (paper_id AND abs_url).
    - contradicting can be empty (the system prompt explicitly allows "no clear contradiction found").
    """
    issues = []
    for i, claim in enumerate(packet.get("claims", []), start=1):
        claim_text = (claim.get("claim") or "")[:60]
        sup = claim.get("supporting", {}) or {}
        if not sup.get("paper_id") or not sup.get("abs_url"):
            issues.append(
                f"Claim {i} ('{claim_text}...'): supporting.paper_id={sup.get('paper_id')!r}, "
                f"supporting.abs_url={sup.get('abs_url')!r}. "
                f"You must actually search for and read a real supporting paper before submitting."
            )
    if issues:
        return ("Packet validation FAILED. Do not resubmit until these are fixed:\n- "
                + "\n- ".join(issues)
                + "\nContinue your research with search_papers + get_paper, then resubmit.")
    return None


def _dispatch_researcher(tool: str, args: dict):
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


def run_researcher(paper_title: str, max_steps: int = 25, verbose: bool = True) -> dict:
    metrics = {
        "wall_clock_s": 0.0,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_errors": 0,
        "completed": False,
        "packet": None,
    }
    start = time.time()
    messages = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": f"/no_think\nPaper title: {paper_title}\n\nGather and submit the packet."},
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
            print(f"  [ollama error: {e}]", flush=True)
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
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Parse error: {e}. Emit valid action JSON."})
            continue

        if verbose:
            print(f"\n── Researcher Step {step} ──────────────")
            print(f"Thought: {thought[:200]}")
            print(f"Action: {tool}({json.dumps(args)[:200]})")

        if tool == "submit_research":
            rejection = _validate_packet(args)
            if rejection is None:
                metrics["completed"] = True
                metrics["packet"] = args
                if verbose:
                    print(f"\nPACKET ACCEPTED: {len(args['claims'])} claims, "
                          f"main paper '{args['main_paper']['title'][:60]}...'")
                break
            metrics["packet_rejections"] = metrics.get("packet_rejections", 0) + 1
            if verbose:
                print(f"\n[VALIDATOR REJECTED packet — researcher must continue]\n{rejection}\n")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation:\n{rejection}"})
            continue

        try:
            observation = _dispatch_researcher(tool, args)
        except Exception as e:
            metrics["tool_errors"] += 1
            observation = f"<tool error: {e}>"

        if verbose:
            preview = observation[:300].replace("\n", " | ")
            print(f"Observation: {preview}{'...' if len(observation) > 300 else ''}")

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    metrics["wall_clock_s"] = round(time.time() - start, 2)
    return metrics


# ============================================================
# Writer
# ============================================================

WRITER_PROMPT = """You are a science writer. You receive a research packet with a main paper and 3 claims.
Each claim has a supporting and a contradicting paper.

Write a ~500-word summary that:
- Opens with the main paper's contribution (1 paragraph)
- For each of the 3 claims: state the claim, then name the supporting paper (with paper_id) and explain why it supports, then the contradicting paper (with paper_id) and explain why it contradicts. If a paper's paper_id is empty, mention "no clear contradiction found".
- Closes with a one-sentence overall assessment.

Emit via final_answer(text, sources) where sources is the list of all non-empty abs_urls from the packet.

You have NO tools beyond final_answer. You cannot fetch more papers. Work with the packet only.

RULES:
- /no_think
- Cite only abs_urls present in the packet. No fabrication.
"""

WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
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
    },
    "required": ["thought", "action"],
    "additionalProperties": False,
}


def run_writer(packet: dict, verbose: bool = True) -> dict:
    metrics = {
        "wall_clock_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "completed": False,
        "final_text": None,
        "sources": [],
    }
    start = time.time()
    packet_json = json.dumps(packet, indent=2)
    messages = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": f"/no_think\nResearch packet:\n```json\n{packet_json}\n```\n\nWrite the summary."},
    ]

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            format=WRITER_SCHEMA,
            options={"temperature": 0.3, "num_ctx": 8192},
            think=False,
        )
    except Exception as e:
        print(f"  [writer ollama error: {e}]", flush=True)
        metrics["wall_clock_s"] = round(time.time() - start, 2)
        return metrics

    metrics["prompt_tokens"] = resp.get("prompt_eval_count", 0) or 0
    metrics["completion_tokens"] = resp.get("eval_count", 0) or 0

    try:
        parsed = json.loads(resp["message"]["content"])
        metrics["completed"] = True
        metrics["final_text"] = parsed["action"]["args"]["text"]
        metrics["sources"] = parsed["action"]["args"]["sources"]
        if verbose:
            print(f"\n── Writer ────────────────────────────")
            print(f"Thought: {parsed['thought'][:200]}")
            print(f"\nFINAL SUMMARY ({len(metrics['final_text'])} chars, {len(metrics['sources'])} sources)")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [writer parse error: {e}]", flush=True)

    metrics["wall_clock_s"] = round(time.time() - start, 2)
    return metrics


# ============================================================
# End-to-end runner
# ============================================================

def run_agent(paper_title: str, max_steps: int = 25, verbose: bool = True) -> dict:
    """Run Researcher → Writer end-to-end. Returns combined metrics matching single_agent.py shape."""
    r = run_researcher(paper_title, max_steps=max_steps, verbose=verbose)
    if not r["completed"]:
        return {
            "paper_title": paper_title,
            "wall_clock_s": r["wall_clock_s"],
            "steps": r["steps"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "tool_errors": r["tool_errors"],
            "completed": False,
            "final_text": None,
            "sources": [],
            "researcher_metrics": r,
            "writer_metrics": None,
        }
    w = run_writer(r["packet"], verbose=verbose)
    return {
        "paper_title": paper_title,
        "wall_clock_s": round(r["wall_clock_s"] + w["wall_clock_s"], 2),
        "steps": r["steps"] + 1,  # +1 for the writer call
        "prompt_tokens": r["prompt_tokens"] + w["prompt_tokens"],
        "completion_tokens": r["completion_tokens"] + w["completion_tokens"],
        "tool_errors": r["tool_errors"],
        "completed": w["completed"],
        "final_text": w["final_text"],
        "sources": w["sources"],
        "researcher_metrics": r,
        "writer_metrics": w,
    }


if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else "Attention Is All You Need"
    m = run_agent(title, max_steps=25, verbose=True)
    print("\n" + "=" * 60)
    print(f"completed:         {m['completed']}")
    print(f"steps:             {m['steps']} (researcher: {m['researcher_metrics']['steps']}, writer: 1)")
    print(f"wall_clock_s:      {m['wall_clock_s']} (researcher: {m['researcher_metrics']['wall_clock_s']}, "
          f"writer: {m['writer_metrics']['wall_clock_s'] if m['writer_metrics'] else 'N/A'})")
    print(f"prompt_tokens:     {m['prompt_tokens']}")
    print(f"completion_tokens: {m['completion_tokens']}")
    print(f"tool_errors:       {m['tool_errors']}")
    print(f"sources:           {m['sources']}")
    print("=" * 60)
    if m["final_text"]:
        print("\nFINAL TEXT:")
        print(m["final_text"])
