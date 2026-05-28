"""
Academic paper search/fetch tools, backed by OpenAlex.
(We switched off arXiv after their rate limiter put us in sticky cooldown.)

OpenAlex: ~100k req/day for unauthenticated users, generally well-behaved.
Abstracts are stored as an inverted index — we reconstruct them to plain text here.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

OPENALEX_API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "trials-course/1.0 (educational; ollama-react-agent)"}

_last_call = 0.0
_MIN_INTERVAL = 0.5  # generous — OpenAlex allows 10 req/s, we use 2


def _throttle():
    global _last_call
    delta = time.time() - _last_call
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_call = time.time()


def _fetch(url: str, max_attempts: int = 4) -> dict:
    """JSON fetch with retry + backoff. Absorbs rate limits at the tool layer."""
    last_err = None
    for attempt in range(max_attempts):
        _throttle()
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                backoff = 5 * (2 ** attempt)
                print(f"  [openalex 429, backing off {backoff}s, attempt {attempt+1}/{max_attempts}]", flush=True)
                time.sleep(backoff)
                continue
            if e.code == 404:
                raise
            backoff = 5 * (2 ** attempt)
            print(f"  [openalex {e.code}, backing off {backoff}s, attempt {attempt+1}/{max_attempts}]", flush=True)
            time.sleep(backoff)
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            backoff = 5 * (2 ** attempt)
            print(f"  [openalex timeout, backing off {backoff}s, attempt {attempt+1}/{max_attempts}]", flush=True)
            time.sleep(backoff)
            continue
    raise RuntimeError(f"openalex unreachable after {max_attempts} attempts: {last_err}")


def _reconstruct_abstract(inv_idx) -> str:
    """OpenAlex stores abstracts as {word: [positions...]}. Flip to text."""
    if not inv_idx:
        return ""
    positions = [(idx, word) for word, indices in inv_idx.items() for idx in indices]
    positions.sort()
    return " ".join(w for _, w in positions)


def _extract_id(work: dict) -> str:
    """OpenAlex IDs come as URLs like 'https://openalex.org/W2741809807' — take the W-ID."""
    url = work.get("id", "")
    return url.rsplit("/", 1)[-1] if url else ""


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Search OpenAlex for academic papers. Returns lightweight metadata records.
    Abstracts are truncated to 250 chars in preview — call get_paper for the full text.
    """
    params = {"search": query, "per-page": max_results}
    url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
    data = _fetch(url)
    results = []
    for work in data.get("results", []):
        wid = _extract_id(work)
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        authorships = work.get("authorships", [])
        authors = [a.get("author", {}).get("display_name", "") for a in authorships[:3]]
        results.append({
            "paper_id": wid,
            "title": work.get("title") or "",
            "authors": [a for a in authors if a],
            "published": (work.get("publication_date") or "")[:10],
            "abstract_preview": abstract[:250] + ("..." if len(abstract) > 250 else ""),
            "abs_url": f"https://openalex.org/{wid}",
            "cited_by_count": work.get("cited_by_count", 0),
        })
    return results


def get_paper(paper_id: str) -> dict:
    """Fetch full metadata + abstract for one OpenAlex work by ID (e.g. 'W2741809807')."""
    url = f"{OPENALEX_API}/{paper_id}"
    try:
        work = _fetch(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"Paper {paper_id} not found"}
        raise
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    authorships = work.get("authorships", [])
    authors = [a.get("author", {}).get("display_name", "") for a in authorships]
    wid = _extract_id(work)
    return {
        "paper_id": wid,
        "title": work.get("title") or "",
        "authors": [a for a in authors if a],
        "published": (work.get("publication_date") or "")[:10],
        "abstract": abstract,
        "abs_url": f"https://openalex.org/{wid}",
        "cited_by_count": work.get("cited_by_count", 0),
    }


if __name__ == "__main__":
    print("=== search_papers('attention is all you need', 3) ===")
    results = search_papers("attention is all you need", 3)
    for r in results:
        print(f"  [{r['paper_id']}] {r['title'][:80]}")
        print(f"    authors: {', '.join(r['authors'])} | {r['published']} | cited {r['cited_by_count']}x")
        print(f"    {r['abstract_preview'][:120]}")
        print()

    print("=== get_paper(top hit) ===")
    if results:
        p = get_paper(results[0]["paper_id"])
        if "error" in p:
            print(f"  ERROR: {p['error']}")
        else:
            print(f"  title:    {p['title']}")
            print(f"  authors:  {', '.join(p['authors'][:3])}")
            print(f"  date:     {p['published']}")
            print(f"  abs_url:  {p['abs_url']}")
            print(f"  abstract: {p['abstract'][:300]}...")
