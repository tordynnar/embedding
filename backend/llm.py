from openai import OpenAI

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "gemma4:e4b"

_client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")


def label_cluster(titles: list[str], fallback: str = "Unlabeled") -> str:
    bullets = "\n".join(f"- {t}" for t in titles)
    prompt = (
        "You are naming a topic cluster. Reply with a 2-4 word topic name only, "
        "no punctuation, no quotes, no explanation.\n\n"
        f"Documents in this cluster:\n{bullets}\n\n"
        "Topic name:"
    )
    try:
        r = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=32,
        )
        text = (r.choices[0].message.content or "").strip()
        first = text.splitlines()[0].strip().strip('"').strip("'.,:")
        if not first or len(first) > 60:
            return fallback
        return first
    except Exception:
        return fallback
