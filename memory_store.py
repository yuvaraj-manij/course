import datetime
import hashlib
import json

import chromadb
import ollama

_client = None
_collection = None
_episodes_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_episodes_collection():
    global _episodes_collection
    _get_collection()  # ensures _client is initialized
    if _episodes_collection is None:
        _episodes_collection = _client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )
    return _episodes_collection


def _embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]


PARAPHRASE_PROMPT = """\
Are these two facts about a user PARAPHRASES of the same fact,
or are they DIFFERENT facts that happen to be phrased similarly?

Fact A: {a}
Fact B: {b}

A "paraphrase" means they convey the exact same information — same entities,
same relationships, same attributes — just worded differently.
"Different" means they share structure or topic but refer to different
entities (different names, different roles, different states).

Answer with exactly one word: "paraphrase" or "different".
"""

_PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["paraphrase", "different"]}},
    "required": ["verdict"],
}


def _check_paraphrase(fact_a: str, fact_b: str) -> str:
    resp = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(a=fact_a, b=fact_b)}],
        format=_PARAPHRASE_SCHEMA,
        options={"temperature": 0, "num_ctx": 2048},
        think=False,
    )
    return json.loads(resp["message"]["content"]).get("verdict", "different")


CONSISTENCY_PROMPT = """\
Determine if two facts about the same user are logically consistent.

Fact A (existing): {a}
Fact B (new):      {b}

Answer with exactly one word: "consistent" or "contradicts".

Consistent means both can be true at the same time about the same person.
Contradicts means they cannot both be true (e.g. different jobs, different
relationships to the same named person, mutually exclusive states).
"""

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["consistent", "contradicts"]}},
    "required": ["verdict"],
}


def _check_consistency(fact_a: str, fact_b: str) -> str:
    resp = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": CONSISTENCY_PROMPT.format(a=fact_a, b=fact_b)}],
        format=_VERDICT_SCHEMA,
        options={"temperature": 0, "num_ctx": 2048},
        think=False,
    )
    return json.loads(resp["message"]["content"]).get("verdict", "consistent")


def add_fact(fact: str, source: str = "user") -> str:
    """
    Stores a fact in the semantic store. Returns the fact's ID.

    Three-zone dedup/contradiction logic:
      distance < 0.15       → duplicate, return existing ID
      0.15 ≤ distance < 0.5 → run consistency check; if contradicts, supersede old fact
      distance ≥ 0.5        → unrelated, just insert

    Metadata: timestamp (ISO 8601), source, version (int), superseded_by, supersedes.
    """
    col = _get_collection()
    embedding = _embed(fact)

    candidates = []
    if col.count() > 0:
        raw = col.query(
            query_embeddings=[embedding],
            n_results=min(col.count(), 9),
            include=["documents", "metadatas", "distances"],
        )
        for fid, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            candidates.append((dist, fid, doc, meta))

    # Pass 1: dedup zone — verify with LLM to avoid swallowing similar-but-distinct facts
    for dist, fid, doc, meta in candidates:
        if dist < 0.15:
            if _check_paraphrase(doc, fact) == "paraphrase":
                return fid  # true duplicate

    # Pass 2: contradiction check in middle zone
    for dist, fid, doc, meta in candidates:
        if 0.15 <= dist < 0.5:
            verdict = _check_consistency(doc, fact)
            if verdict == "contradicts":
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                new_id = f"{hashlib.sha256(fact.encode()).hexdigest()[:8]}-{ts}"
                old_version = int(meta.get("version", 1))

                col.add(
                    ids=[new_id],
                    documents=[fact],
                    embeddings=[embedding],
                    metadatas=[{
                        "timestamp": ts,
                        "source": source,
                        "version": old_version + 1,
                        "superseded_by": "",
                        "supersedes": fid,
                    }],
                )
                col.update(
                    ids=[fid],
                    metadatas=[{**meta, "superseded_by": new_id}],
                )
                return new_id

    # Pass 3: no duplicate, no contradiction — simple insert
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_id = f"{hashlib.sha256(fact.encode()).hexdigest()[:8]}-{ts}"
    col.add(
        ids=[new_id],
        documents=[fact],
        embeddings=[embedding],
        metadatas=[{
            "timestamp": ts,
            "source": source,
            "version": 1,
            "superseded_by": "",
            "supersedes": "",
        }],
    )
    return new_id


def recall(
    query: str,
    k: int = 5,
    threshold: float = 0.5,
    include_superseded: bool = False,
) -> list[dict]:
    """
    Searches the semantic store for facts matching the query.
    Returns up to k results, each a dict:
        {"fact": str, "distance": float, "timestamp": str, "version": int}
    Superseded facts are excluded by default.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    raw = col.query(
        query_embeddings=[_embed(query)],
        n_results=min(col.count(), k * 3),
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for doc, meta, dist in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        if dist >= threshold:
            continue
        if not include_superseded and meta.get("superseded_by"):
            continue
        results.append({
            "fact": doc,
            "distance": dist,
            "timestamp": meta.get("timestamp", ""),
            "version": meta.get("version", 1),
        })
        if len(results) >= k:
            break
    return results


def list_facts() -> list[dict]:
    """Return all stored facts with their metadata. For debugging only."""
    col = _get_collection()
    if col.count() == 0:
        return []
    data = col.get(include=["documents", "metadatas"])
    return [
        {"id": fid, "fact": doc, **meta}
        for fid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def recall_all_facts(limit: int = 20) -> list[dict]:
    """Return the most recent N facts in the store, newest first.
    Excludes superseded facts. No similarity filtering."""
    col = _get_collection()
    if col.count() == 0:
        return []
    data = col.get(include=["documents", "metadatas"])
    facts = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if meta.get("superseded_by"):
            continue
        facts.append({
            "fact": doc,
            "timestamp": meta.get("timestamp", ""),
            "version": meta.get("version", 1),
        })
    facts.sort(key=lambda x: x["timestamp"], reverse=True)
    return facts[:limit]


def add_episodic(session_summary: str, session_id: str, started_at: str, ended_at: str) -> str:
    """Stores a session-level summary in the episodes collection. Returns the session_id."""
    col = _get_episodes_collection()
    col.add(
        ids=[session_id],
        documents=[session_summary],
        embeddings=[_embed(session_summary)],
        metadatas=[{"started_at": started_at, "ended_at": ended_at}],
    )
    return session_id


def recall_episodic(query: str, k: int = 3) -> list[dict]:
    """Searches episodic memory by similarity.
    Returns [{"summary": str, "started_at": str, "ended_at": str, "distance": float}]
    """
    col = _get_episodes_collection()
    if col.count() == 0:
        return []
    raw = col.query(
        query_embeddings=[_embed(query)],
        n_results=min(col.count(), k),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "summary": doc,
            "started_at": meta.get("started_at", ""),
            "ended_at": meta.get("ended_at", ""),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]


def forget(fact_id: str) -> bool:
    """Remove a fact by ID. Returns True if deleted."""
    col = _get_collection()
    existing = col.get(ids=[fact_id])
    if not existing["ids"]:
        return False
    col.delete(ids=[fact_id])
    return True


if __name__ == "__main__":
    # Start clean
    import shutil
    shutil.rmtree("./chroma_db", ignore_errors=True)

    # Add some facts
    id1 = add_fact("The user has a daughter named Ava.")
    id2 = add_fact("The user's dog is a beagle named Bruno.")
    id3 = add_fact("The user works at LifeBridge Insurance.")
    print(f"Added: {id1}, {id2}, {id3}")

    # Dedup test: re-add a rephrasing — should return same ID, not create new
    id1b = add_fact("Ava is the user's daughter.")
    print(f"Dedup test: {id1} vs {id1b}  (should be same)")
    print(f"Total facts: {len(list_facts())}  (should be 3, not 4)")

    # Strong-match recall
    hits = recall("Does the user have children?")
    print(f"\nStrong match — children:")
    for h in hits:
        print(f"  [{h['distance']:.3f}] {h['fact']}")

    # No-match recall — should return empty list
    hits = recall("What is the user's favorite color?")
    print(f"\nNo-match — favorite color:")
    print(f"  results: {hits}  (should be empty)")

    # Weak-match recall — also should return empty if below threshold
    hits = recall("Tell me about user's hobbies.")
    print(f"\nWeak match — hobbies:")
    print(f"  results: {hits}  (should be empty, since no hobby facts exist)")
