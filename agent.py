import ast
import operator
import re

import httpx
import trafilatura
from ddgs import DDGS


def web_search(query: str) -> str:
    results = list(DDGS().text(query, max_results=5))
    if not results:
        return f"No results for {query!r}"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

    output = "\n".join(lines)
    if len(output) > 1500:
        output = output[:1500] + "..."
    return output


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def read_url(url: str) -> str:
    response = httpx.get(
        url, timeout=10, follow_redirects=True, headers={"User-Agent": _UA}
    )
    text = trafilatura.extract(response.text, favor_recall=True)
    if text is None:
        text = _strip_html(response.text)
    if not text:
        return f"No extractable content from {url}"
    if len(text) > 3000:
        text = text[:3000] + "..."
    return text


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError("operator not allowed")
        return _ALLOWED_BINOPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError("node not allowed")


def calculator(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree)
    except Exception:
        return "Error: unsafe expression"
    return str(result)


if __name__ == "__main__":                                                                                            
      print(web_search("LangGraph github stars"))
      print("---")                                                                                                      
      print(read_url("https://github.com/langchain-ai/langgraph"))
      print("---")                                                                                                      
      print(calculator("17**7 + 1889"))
      print(calculator("__import__('os')"))  # should reject  
