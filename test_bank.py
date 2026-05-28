"""Phase 3b: 20-query test bank.

Runs every query through the schema-constrained agent and records
tool-call validity per step. Schema violations should be 0 — if any
appear, the schema or Ollama's structured-output path has a bug.

Results written to test_bank_results.json.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field

import ollama

from pypi_loop import TOOLS, _action_key, _fmt_args, _truncate
from pypi_react import ACTION_SCHEMA, SYSTEM_PROMPT

QUERIES = [
    # ── Easy lookups (single tool, 1–2 steps) ──────────────────────────
    "What's the latest version of pydantic?",
    "Who maintains the requests package — give me the repo URL.",
    "What Python versions does numpy support?",
    "How many dependencies does fastapi have?",
    "When was the latest release of django?",

    # ── Comparison (forces compare_packages) ───────────────────────────
    "Compare flask vs fastapi — which has more dependencies?",
    "Compare numpy vs scipy on version and Python requirement.",
    "Compare pydantic vs attrs — version and dependency count.",
    "Compare httpx and requests — which is newer?",

    # ── README-required (forces read_github_readme) ────────────────────
    "What does the rich library do, according to its README?",
    "What's the headline feature of typer based on its README?",
    "Summarize what polars is for, from its repo README.",

    # ── Search-first (no obvious package name) ─────────────────────────
    "Find me a Python library for parsing PDFs.",
    "What's a popular Python library for working with time zones?",
    "Find a Python package for generating fake data for tests.",

    # ── Adversarial / edge cases ───────────────────────────────────────
    "Tell me about the package 'asdf-not-real-pkg-xyz' on PyPI.",
    "What's the version of foobarbaz12345?",
    "Compare httpx and asdf-not-real-pkg — which is newer?",
    "Find me a non-existent fake imaginary library.",

    # ── Out of scope ───────────────────────────────────────────────────
    "What's the weather in Paris?",
]

assert len(QUERIES) == 20


@dataclass
class RunResult:
    query: str
    steps_used: int = 0
    final_answer: str = ""
    sources: list[str] = field(default_factory=list)
    schema_violations: int = 0
    tool_call_validity_per_step: list[bool] = field(default_factory=list)
    completed: bool = False


def run_instrumented(query: str, max_steps: int = 8) -> RunResult:
    result = RunResult(query=query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    seen_actions: dict = {}
    last_thought = ""

    for step in range(1, max_steps + 1):
        result.steps_used = step
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
            valid = obj["tool"] in {*TOOLS, "final_answer"}
        except (json.JSONDecodeError, KeyError):
            result.schema_violations += 1
            result.tool_call_validity_per_step.append(False)
            obs = (
                "Error: your last output was malformed. "
                "Re-emit valid JSON matching the schema."
            )
            print("Thought: (unparseable)")
            print(f"Observation: {obs}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        if not valid:
            result.schema_violations += 1
        result.tool_call_validity_per_step.append(valid)

        last_thought = obj["thought"]
        tool = obj["tool"]
        args = obj["args"]

        print(f"Thought: {obj['thought']}")
        print(f"Action: {tool}({_fmt_args(args)})")

        if tool == "final_answer":
            result.final_answer = args.get("text", "")
            result.sources = args.get("sources", [])
            result.completed = True
            print("Observation: <final_answer returned>")
            break

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

    if not result.completed:
        result.final_answer = f"FAILED: step limit reached. Last thought: {last_thought}"

    return result


def print_summary(results: list[RunResult]) -> None:
    total_violations = sum(r.schema_violations for r in results)
    total_turns = sum(r.steps_used for r in results)
    completed = sum(1 for r in results if r.completed)
    hit_cap = sum(1 for r in results if not r.completed)
    step_counts = [r.steps_used for r in results]

    print("\n" + "=" * 50)
    print("=== 20-query test bank ===")
    print(f"Total queries:                {len(results)}")
    print(f"Schema violations (any step): {total_violations} / {total_turns} total turns")
    print(f"Queries completed in ≤ 8 steps: {completed} / {len(results)}")
    print(f"Queries that hit final_answer:   {completed} / {len(results)}")
    print(f"Queries that hit step cap:       {hit_cap} / {len(results)}")
    print()
    print(f"Per-query step counts: {step_counts}")
    print(
        f"Average steps: {statistics.mean(step_counts):.1f}  "
        f"Min: {min(step_counts)}  Max: {max(step_counts)}"
    )

    print("\nPer-query results:")
    for i, r in enumerate(results, 1):
        status = "✓" if r.completed else "✗ (cap)"
        viol = f" [{r.schema_violations} violations]" if r.schema_violations else ""
        print(f"  Q{i:02d} [{r.steps_used} steps] {status}{viol}  {r.query[:60]}")


if __name__ == "__main__":
    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'=' * 70}")
        print(f"QUERY {i:02d}/20: {q}")
        print("=" * 70)
        r = run_instrumented(q)
        results.append(r)
        print(f"\nFINAL ANSWER: {r.final_answer[:200]}")
        if r.sources:
            print(f"SOURCES: {r.sources}")

    out_path = "test_bank_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults written to {out_path}")

    print_summary(results)
