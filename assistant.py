import datetime
import json
import uuid

import ollama
import tiktoken

from memory_store import (
    add_episodic,
    add_fact,
    recall as _store_recall,
    recall_all_facts as _store_recall_all,
    recall_episodic as _store_recall_episodic,
)


def _branch(tool_name: str, args_schema: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {"type": "string", "const": tool_name},
            "args": args_schema,
        },
        "required": ["thought", "tool", "args"],
    }


ACTION_SCHEMA = {
    "type": "object",
    "oneOf": [
        _branch("recall", {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }),
        _branch("recall_all_facts", {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": [],
        }),
        _branch("recall_episodic", {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }),
        _branch("respond", {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }),
    ],
}


SYSTEM_PROMPT = """\
You are a personal assistant with persistent memory across conversations.

You have four tools:
- recall(query) — searches long-term semantic memory for facts matching a query. \
Use for specific lookups ("what's my dog's name?").
- recall_all_facts(limit) — returns the most recent facts about the user with no \
similarity filtering. Use for broad questions like "what do you know about me?" \
or "tell me what you remember."
- recall_episodic(query) — searches summaries of past conversations by similarity. \
Use when the user references an earlier session ("remember when we talked about...?") \
or asks about themes spanning multiple conversations.
- respond(text) — your reply to the user. The conversation ends this turn after \
you respond.

When to pick which memory tool:
- Specific lookup ("what's my dog's name?") → recall(query)
- Broad inventory ("what do you know about me?") → recall_all_facts
- Cross-session reference ("you mentioned X last time") → recall_episodic(query)

When NOT to call any recall tool:
- The user is stating a NEW fact for you to remember. Just acknowledge — \
fact extraction is handled separately.
- The user is making small talk that doesn't reference prior context.

Honesty rules:
- If recall returns "no relevant facts found", say "I don't know" or \
"you haven't told me that yet" rather than guessing.
- Never claim to remember something that isn't in your recall results.
"""


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

EXTRACT_PROMPT = """\
Extract stable facts about the user from their message.

A fact is a stable, declarative statement about the user — their relationships, \
preferences, circumstances, or attributes.

Examples of facts (include these):
- "I have a sister named Mia" → "The user has a sister named Mia."
- "My dog is a beagle" → "The user's dog is a beagle."
- "I love hiking" → "The user enjoys hiking."

Examples of non-facts (exclude these):
- "I'm hungry" — transient state, not stable
- "What's the weather?" — a question, not a statement about the user
- "Thanks!" — small talk with no factual content

Rules:
- Canonicalize every fact to third-person form starting with "The user".
- Only include facts that are stable across time.
- If the message contains no facts about the user, return {"facts": []}.
- Do not invent or infer beyond what is explicitly stated.
"""


def extract_facts(user_msg: str) -> list[str]:
    """
    Calls the LLM with a fact-extraction prompt.
    Returns a list of canonicalized facts about the user.
    Empty list if the message contains no factual statements.
    """
    resp = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        format=EXTRACT_SCHEMA,
        options={"temperature": 0, "num_ctx": 2048},
        think=False,
    )
    return json.loads(resp["message"]["content"]).get("facts", [])


TOKEN_CAP = 6000
KEEP_RECENT_TURNS = 4
TRIGGER_AT = 0.85

_enc = tiktoken.get_encoding("cl100k_base")

SUMMARY_PROMPT = """\
You are summarizing the earlier portion of a conversation between a user and an
assistant so it can be compressed into limited context space.

Produce a structured summary in this format:

FACTS STATED BY USER:
- <list each durable fact about the user — relationships, preferences,
  circumstances, attributes — as a separate bullet>

DECISIONS / COMMITMENTS:
- <anything the user or assistant decided to do, recommendations made,
  promises kept or pending>

OPEN QUESTIONS:
- <anything the user asked that wasn't fully answered, or things deferred>

Rules:
- Preserve specific names, numbers, dates verbatim.
- Drop pleasantries, acknowledgments, and small talk.
- If a section has no content, write "(none)".
- Keep the whole summary under 500 tokens.
"""


def count_messages_tokens(messages: list[dict]) -> int:
    """Token total across all message contents."""
    return sum(len(_enc.encode(m["content"])) for m in messages)


def summarize_old_turns(old_messages: list[dict]) -> str:
    """Compress the older portion of conversation into a single summary string."""
    resp = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in old_messages
            )},
        ],
        options={"temperature": 0, "num_ctx": 4096},
        think=False,
    )
    return resp["message"]["content"]


def maybe_summarize(conversation: list[dict]) -> list[dict]:
    """
    If the conversation exceeds TRIGGER_AT * TOKEN_CAP, summarize all but
    the last KEEP_RECENT_TURNS pairs. Returns the (possibly summarized) list.
    """
    total = count_messages_tokens(conversation)
    if total < TRIGGER_AT * TOKEN_CAP:
        return conversation

    keep = KEEP_RECENT_TURNS * 2          # pairs → individual messages
    cutoff = max(0, len(conversation) - keep)
    old_messages = conversation[:cutoff]
    recent = conversation[cutoff:]

    if not old_messages:
        return conversation

    summary_text = summarize_old_turns(old_messages)
    summary_tok = len(_enc.encode(summary_text))
    new_ctx = count_messages_tokens(recent) + summary_tok
    print(
        f"[summary] context={total} tok → summarized {len(old_messages)} old messages "
        f"into ~{summary_tok} tok summary → new context={new_ctx} tok"
    )
    print(f"[summary text]\n{summary_text}\n[/summary text]")

    return [{"role": "system", "content": f"Summary of earlier conversation:\n{summary_text}"}] + recent


def recall(query: str) -> str:
    hits = _store_recall(query)
    if not hits:
        return "no relevant facts found"
    return "\n".join(f"[{h['distance']:.3f}] {h['fact']}" for h in hits)


def recall_all_facts(limit: int = 20) -> str:
    facts = _store_recall_all(limit)
    if not facts:
        return "no facts stored yet"
    return "\n".join(f"[v{f['version']}] {f['fact']}" for f in facts)


def recall_episodic(query: str) -> str:
    hits = _store_recall_episodic(query)
    if not hits:
        return "no past sessions found"
    return "\n\n".join(
        f"[session {h['started_at'][:10]}]\n{h['summary']}" for h in hits
    )


def respond(text: str) -> str:
    return text


def chat_turn(user_msg: str, conversation: list) -> str:
    facts = extract_facts(user_msg)
    for fact in facts:
        add_fact(fact)
        print(f"  [stored] {fact}")

    conversation.append({"role": "user", "content": user_msg})

    for _ in range(5):
        conversation[:] = maybe_summarize(conversation)
        resp = ollama.chat(
            model="qwen3:14b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            format=ACTION_SCHEMA,
            options={"temperature": 0.2, "num_ctx": 4096},
            think=False,
        )
        raw = resp["message"]["content"]
        obj = json.loads(raw)
        tool = obj["tool"]
        args = obj["args"]

        print(f"  [tool={tool}] args={args}")

        if tool == "respond":
            reply = respond(args["text"])
            conversation.append({"role": "assistant", "content": reply})
            return reply

        if tool == "recall":
            obs = recall(args["query"])
            print(f"  [recall result] {obs}")
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": f"Memory: {obs}"})

        elif tool == "recall_all_facts":
            obs = recall_all_facts(args.get("limit", 20))
            print(f"  [recall_all_facts result] {obs[:200]}")
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": f"Memory: {obs}"})

        elif tool == "recall_episodic":
            obs = recall_episodic(args["query"])
            print(f"  [recall_episodic result] {obs[:200]}")
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": f"Memory: {obs}"})

    return "I'm having trouble responding."


if __name__ == "__main__":
    conversation = []
    session_start = datetime.datetime.now(datetime.timezone.utc)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", ":q"}:
            if conversation:
                full_summary = summarize_old_turns(conversation)
                add_episodic(
                    session_summary=full_summary,
                    session_id=str(uuid.uuid4()),
                    started_at=session_start.isoformat(),
                    ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                print(f"[episodic] stored session summary ({len(full_summary)} chars)")
            break
        reply = chat_turn(user_input, conversation)
        print(f"\nAssistant: {reply}")
