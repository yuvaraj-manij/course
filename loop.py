import ollama

from agent import calculator, read_url, web_search
from react import SYSTEM_PROMPT, MalformedAction, parse_action

TOOLS = {
    "web_search": web_search,
    "read_url": read_url,
    "calculator": calculator,
}


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _truncate(text: str, limit: int = 500) -> str:
    text = str(text)
    if len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text)} chars total]"
    return text


def _action_key(tool: str, args: dict):
    try:
        return (tool, frozenset(args.items()))
    except TypeError:
        return (tool, repr(sorted(args.items())))


def run_agent(question: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    seen_actions: dict = {}
    last_thought = ""

    for step in range(1, max_steps + 1):
        print(f"\n── Step {step} ──────────────────────")

        resp = ollama.chat(
            model="qwen3:14b",
            messages=messages,
            format="json",
            options={"temperature": 0.2},
        )
        raw = resp["message"]["content"]
        messages.append({"role": "assistant", "content": raw})

        try:
            action = parse_action(raw)
        except MalformedAction as e:
            obs = (
                f"Error: your last output was malformed: {e}. "
                "Re-emit valid JSON matching the schema."
            )
            print("Thought: (unparseable)")
            print(f"Action: <malformed> {_truncate(raw, 200)}")
            print(f"Observation: {_truncate(obs)}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        last_thought = action["thought"]
        tool = action["tool"]
        args = action["args"]

        print(f"Thought: {action['thought']}")
        print(f"Action: {tool}({_fmt_args(args)})")

        if action["is_final"]:
            answer = args.get("text", "")
            print(f"Observation: <final_answer returned>")
            return answer

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

    return f"FAILED: step limit reached. Last thought: {last_thought}"


if __name__ == "__main__":
    tasks = [
        "What is the population of the capital of the country that won "
        "the most recent FIFA World Cup, divided by 1000?",
        "Find the GitHub stars of the LangGraph repo and compare to CrewAI.",
        "What is 17^7 + the year the Eiffel Tower was completed?",
    ]
    for t in tasks:
        print("=" * 70)
        print("QUESTION:", t)
        print("=" * 70)
        answer = run_agent(t)
        print("\nFINAL ANSWER:", answer, "\n")
