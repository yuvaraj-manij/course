"""
Chapter 6 — Blackboard state store.

state.json is the shared handoff contract between Researcher, Drafter, Editor.
Each stage reads its predecessor's field and writes its own. The store does
atomic writes (tmp + rename) so a crash mid-write cannot corrupt the file.

Validators are receiver-side: before an agent runs, the coordinator checks the
predecessor's field meets the contract. This is the Ch 5 retrofit — without it,
agents can satisfy the schema with empty payloads.
"""
import json
import os
import tempfile
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_state(topic: str) -> dict:
    return {
        "topic": topic,
        "research": None,
        "draft": None,
        "final": None,
        "history": [],
    }


def load_state(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(state: dict, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".state_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def log_event(state: dict, event: str, **kwargs) -> None:
    state["history"].append({"at": now_iso(), "event": event, **kwargs})


def validate_research(research: dict | None) -> str | None:
    if not research:
        return "research field is empty"
    claims = research.get("claims") or []
    if len(claims) < 3:
        return f"research must have >=3 claims, got {len(claims)}"
    for i, c in enumerate(claims, 1):
        text = (c.get("text") or "").strip()
        url = (c.get("url") or "").strip()
        if not text:
            return f"claim {i} has empty text"
        if not url:
            return f"claim {i} has empty url"
        if not url.startswith("http"):
            return f"claim {i} url does not look like a URL: {url!r}"
    return None


def validate_draft(draft: dict | None, research: dict) -> str | None:
    if not draft:
        return "draft field is empty"
    body = (draft.get("body") or "").strip()
    if not body:
        return "draft.body is empty"
    word_count = len(body.split())
    if word_count < 500:
        return f"draft.body has {word_count} words, need >=500"
    title = (draft.get("title") or "").strip()
    if not title:
        return "draft.title is empty"
    citations = draft.get("citations_used") or []
    if not citations:
        return "draft.citations_used is empty"
    valid_urls = {(c.get("url") or "").strip() for c in research.get("claims", [])}
    for cite in citations:
        if cite not in valid_urls:
            return f"draft cites {cite!r} which is not in research.claims URLs"
        if cite not in body:
            return f"draft.citations_used lists {cite!r} but the URL does not appear in draft.body"
    return None


def validate_final(final: dict | None) -> str | None:
    if not final:
        return "final field is empty"
    body = (final.get("body") or "").strip()
    if not body:
        return "final.body is empty"
    word_count = len(body.split())
    if word_count < 500:
        return f"final.body has {word_count} words, need >=500"
    edits = final.get("edits_made") or []
    if not edits:
        return "final.edits_made is empty (editor must report at least one change)"
    return None


def next_stage(state: dict) -> str | None:
    if state["research"] is None:
        return "research"
    if state["draft"] is None:
        return "draft"
    if state["final"] is None:
        return "final"
    return None


if __name__ == "__main__":
    s = init_state("test topic")
    print("init:", json.dumps(s, indent=2))
    print("next:", next_stage(s))
    print("validate empty research:", validate_research(s["research"]))
