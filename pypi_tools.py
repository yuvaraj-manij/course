import re

import httpx
from ddgs import DDGS
from urllib.parse import urlparse


def search_pypi(query: str) -> str:
    results = list(DDGS().text(f"{query} site:pypi.org/project", max_results=10))
    if not results:
        return f"No PyPI results for {query!r}"

    lines = []
    for i, r in enumerate(results, 1):
        url = r.get("href", "")
        snippet = r.get("body", "")

        # extract package name from https://pypi.org/project/<name>/
        parts = [p for p in url.split("/") if p]
        try:
            name = parts[parts.index("project") + 1]
        except (ValueError, IndexError):
            name = url

        lines.append(f"{i}. {name} — {snippet}")

    output = "\n".join(lines)
    if len(output) > 2000:
        output = output[:2000] + "..."
    return output


def get_package_info(name: str) -> str:
    resp = httpx.get(f"https://pypi.org/pypi/{name}/json", timeout=10)
    if resp.status_code == 404:
        return f"Package {name!r} not found on PyPI"
    if resp.status_code != 200:
        return f"PyPI returned {resp.status_code} for {name!r}"

    info = resp.json()["info"]
    urls = resp.json().get("urls", [])

    version = info.get("version", "unknown")
    summary = info.get("summary") or "no summary"
    requires_python = info.get("requires_python") or "not specified"

    release_date = "unknown"
    if urls:
        release_date = max(urls, key=lambda f: f.get("upload_time", ""))["upload_time"]

    # find a repo URL from project_urls
    project_urls = info.get("project_urls") or {}
    repo_url = "<none found>"
    for key in ("Source", "Source Code", "Repository", "Homepage", "GitHub"):
        candidate = project_urls.get(key)
        if candidate and "github.com" in candidate:
            repo_url = candidate
            break
    else:
        # fallback: any value that is a bare github.com/<owner>/<repo> URL
        for v in project_urls.values():
            if isinstance(v, str) and re.match(
                r"^https?://github\.com/[^/]+/[^/]+/?$", v
            ):
                repo_url = v
                break

    deps = info.get("requires_dist") or []
    if deps:
        shown = deps[:10]
        dep_lines = "\n".join(f"  - {d}" for d in shown)
        if len(deps) > 10:
            dep_lines += f"\n  ... and {len(deps) - 10} more"
    else:
        dep_lines = "  (none listed)"

    return (
        f"name:             {name}\n"
        f"version:          {version}\n"
        f"summary:          {summary}\n"
        f"release_date:     {release_date}\n"
        f"requires_python:  {requires_python}\n"
        f"repo_url:         {repo_url}\n"
        f"dependencies ({len(deps)}):\n{dep_lines}"
    )


def read_github_readme(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    if "github.com" not in parsed.netloc:
        return "Error: not a github.com URL"

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return "Error: not a github.com URL"

    owner, repo = parts[0], parts[1].removesuffix(".git")
    base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"

    for filename in ("README.md", "README.rst", "README.txt"):
        try:
            resp = httpx.get(f"{base}/{filename}", timeout=10, follow_redirects=True)
        except Exception as e:
            return f"Error fetching README: {e}"
        if resp.status_code == 200:
            text = resp.text
            if len(text) > 12000:
                text = text[:12000] + "..."
            return text

    return f"No README found at {repo_url}"


def _parse_info(info_str: str) -> dict:
    """Parse get_package_info output into a dict. Returns {'error': ...} on failure."""
    if info_str.startswith("Package") or info_str.startswith("PyPI returned"):
        return {"error": info_str}
    result = {}
    for line in info_str.splitlines():
        line = line.strip()
        for key in ("name", "version", "summary", "release_date",
                    "requires_python", "repo_url"):
            if line.startswith(f"{key}:"):
                result[key] = line.split(":", 1)[1].strip()
                break
        if line.startswith("dependencies ("):
            try:
                result["deps_count"] = int(line.split("(")[1].split(")")[0])
            except (IndexError, ValueError):
                result["deps_count"] = 0
    return result


def compare_packages(name_a: str, name_b: str) -> str:
    info_a = _parse_info(get_package_info(name_a))
    info_b = _parse_info(get_package_info(name_b))

    header = f"Comparison: {name_a} vs {name_b}"

    if "error" in info_a or "error" in info_b:
        lines = [header, ""]
        if "error" in info_a:
            lines.append(f"ERROR for {name_a}: {info_a['error']}")
        if "error" in info_b:
            lines.append(f"ERROR for {name_b}: {info_b['error']}")
        return "\n".join(lines)

    def row(label, ka, kb):
        va = str(info_a.get(ka, "unknown"))
        vb = str(info_b.get(kb, "unknown"))
        return f"{label:<16} {va:<30} | {vb}"

    # trim ISO timestamps to date only
    for d in (info_a, info_b):
        if "release_date" in d:
            d["release_date"] = d["release_date"][:10]

    return "\n".join([
        header,
        "",
        row("Version:",      "version",        "version"),
        row("Released:",     "release_date",   "release_date"),
        row("Python req:",   "requires_python","requires_python"),
        row("Deps count:",   "deps_count",     "deps_count"),
        row("Repo:",         "repo_url",        "repo_url"),
        "",
        f"Summary {name_a}: {info_a.get('summary', 'n/a')}",
        f"Summary {name_b}: {info_b.get('summary', 'n/a')}",
    ])


if __name__ == "__main__":
    print("=== search_pypi ===")
    print(search_pypi("http client async"))
    print("\n=== get_package_info(httpx) ===")
    print(get_package_info("httpx"))
    print("\n=== get_package_info(asdf-not-a-pkg) ===")
    print(get_package_info("asdf-not-a-pkg-asdf"))
    print("\n=== read_github_readme(httpx) ===")
    # use the repo URL you get from get_package_info("httpx")
    print(read_github_readme("https://github.com/encode/httpx")[:500])
    print("\n=== compare_packages(httpx, requests) ===")
    print(compare_packages("httpx", "requests"))

