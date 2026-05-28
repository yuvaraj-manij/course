"""
Chapter 6 — Blackboard coordinator.

Deterministic while-loop: reads state, dispatches the right agent, validates
output against the handoff schema, saves to disk. If validation fails, re-invokes
the agent with the error as feedback (up to N retries per stage).

Resume-from-disk: if state.json exists from a previous run, picks up where it
left off. Kill mid-run, restart — completed stages are skipped.
"""
import argparse
import os
import time

from state_store import (
    init_state, load_state, save_state, log_event, next_stage,
    validate_research, validate_draft, validate_final,
)
from blog_agents import run_researcher, run_drafter, run_editor

DEFAULT_TOPIC = "How does ReAct prompting differ from chain-of-thought?"
DEFAULT_STATE_PATH = "blog_state.json"
MAX_RETRIES_PER_STAGE = 2


def _run_with_retry(stage_name, run_fn, validator, state, totals, state_path, verbose):
    """Run a single-shot agent with validator-driven retry. Returns (ok, error)."""
    field = stage_name  # "draft" or "final"
    retry_feedback = None
    for attempt in range(MAX_RETRIES_PER_STAGE + 1):
        result = run_fn(state, retry_feedback=retry_feedback, verbose=verbose)
        produced = result.get(field)
        if produced is None:
            log_event(state, f"{stage_name}_agent_failed", attempt=attempt + 1, error=result.get("error"))
            save_state(state, state_path)
            return False, result.get("error", "no output")
        totals["prompt_tokens"] += result["metrics"]["prompt_tokens"]
        totals["completion_tokens"] += result["metrics"]["completion_tokens"]

        err = validator(produced, state)
        if err is None:
            state[field] = produced
            log_event(state, f"{stage_name}_completed", attempt=attempt + 1)
            save_state(state, state_path)
            if verbose:
                wc = len(produced["body"].split())
                extra = (f"{len(produced.get('citations_used', []))} citations"
                         if stage_name == "draft"
                         else f"{len(produced.get('edits_made', []))} edits")
                print(f"[coord] {stage_name} saved (attempt {attempt + 1}): {wc} words, {extra}")
            return True, None

        if verbose:
            print(f"[coord] {stage_name} validation failed (attempt {attempt + 1}): {err}")
        log_event(state, f"{stage_name}_validation_failed", attempt=attempt + 1, error=err)
        retry_feedback = err

    return False, f"max retries exceeded; last error: {retry_feedback}"


def _validate_draft_with_state(draft, state):
    return validate_draft(draft, state["research"])


def _validate_final_with_state(final, state):
    err = validate_final(final)
    if err:
        return err
    missing = [u for u in state["draft"]["citations_used"] if u not in final["body"]]
    if missing:
        return f"editor removed citations from body: {missing}"
    return None


def coordinate(topic: str, state_path: str, verbose: bool = True) -> dict:
    state = load_state(state_path)
    if state is None:
        state = init_state(topic)
        log_event(state, "init", topic=topic)
        save_state(state, state_path)
        if verbose:
            print(f"[coord] fresh state at {state_path} for topic: {topic!r}")
    else:
        if state["topic"] != topic:
            print(f"[coord] WARNING: state on disk is for topic {state['topic']!r}, "
                  f"requested {topic!r}. Resuming existing state. Use --reset to start over.")
        if verbose:
            print(f"[coord] resuming from {state_path}: "
                  f"research={'done' if state['research'] else 'pending'} "
                  f"draft={'done' if state['draft'] else 'pending'} "
                  f"final={'done' if state['final'] else 'pending'}")

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "researcher_steps": 0}
    start = time.time()

    while True:
        stage = next_stage(state)
        if stage is None:
            break

        if verbose:
            print(f"\n[coord] stage = {stage}")

        if stage == "research":
            result = run_researcher(state, max_steps=15, verbose=verbose)
            if result.get("research") is None:
                log_event(state, "research_agent_failed", error=result.get("error"))
                save_state(state, state_path)
                return {"state": state, "totals": totals,
                        "wall_clock_s": round(time.time() - start, 2), "completed": False,
                        "fail_reason": result.get("error")}
            err = validate_research(result["research"])
            if err:
                log_event(state, "research_validation_failed", error=err)
                save_state(state, state_path)
                return {"state": state, "totals": totals,
                        "wall_clock_s": round(time.time() - start, 2), "completed": False,
                        "fail_reason": f"research validation: {err}"}
            state["research"] = result["research"]
            totals["prompt_tokens"] += result["metrics"]["prompt_tokens"]
            totals["completion_tokens"] += result["metrics"]["completion_tokens"]
            totals["researcher_steps"] = result["metrics"]["steps"]
            log_event(state, "research_completed",
                      claims=len(result["research"]["claims"]),
                      opened_urls=len(result["research"]["opened_urls"]))
            save_state(state, state_path)
            if verbose:
                print(f"[coord] research saved: {len(result['research']['claims'])} claims, "
                      f"{len(result['research']['opened_urls'])} URLs opened")

        elif stage == "draft":
            ok, err = _run_with_retry("draft", run_drafter, _validate_draft_with_state,
                                       state, totals, state_path, verbose)
            if not ok:
                return {"state": state, "totals": totals,
                        "wall_clock_s": round(time.time() - start, 2), "completed": False,
                        "fail_reason": f"draft: {err}"}

        elif stage == "final":
            ok, err = _run_with_retry("final", run_editor, _validate_final_with_state,
                                       state, totals, state_path, verbose)
            if not ok:
                return {"state": state, "totals": totals,
                        "wall_clock_s": round(time.time() - start, 2), "completed": False,
                        "fail_reason": f"final: {err}"}

    return {"state": state, "totals": totals,
            "wall_clock_s": round(time.time() - start, 2), "completed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    parser.add_argument("--reset", action="store_true", help="Delete existing state and start fresh")
    args = parser.parse_args()

    if args.reset and os.path.exists(args.state):
        os.remove(args.state)
        print(f"[main] deleted {args.state}")

    result = coordinate(args.topic, args.state, verbose=True)

    state = result["state"]
    totals = result["totals"]
    print("\n" + "=" * 60)
    print(f"completed:          {result['completed']}")
    if not result["completed"]:
        print(f"fail_reason:        {result.get('fail_reason')}")
    print(f"wall_clock_s:       {result['wall_clock_s']}")
    print(f"prompt_tokens:      {totals['prompt_tokens']}")
    print(f"completion_tokens:  {totals['completion_tokens']}")
    print(f"researcher_steps:   {totals['researcher_steps']}")
    if state["research"]:
        print(f"claims:             {len(state['research']['claims'])}")
        print(f"opened_urls:        {len(state['research']['opened_urls'])}")
    if state["final"]:
        body = state["final"]["body"]
        print(f"final word count:   {len(body.split())}")
        print(f"edits made:         {len(state['final']['edits_made'])}")
    print("=" * 60)

    if state["final"]:
        print(f"\n# {state['final']['title']}\n")
        print(state["final"]["body"])
        print("\n--- EDITS MADE ---")
        for e in state["final"]["edits_made"]:
            print(f"- {e}")


if __name__ == "__main__":
    main()
