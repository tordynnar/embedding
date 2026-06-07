import httpx

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "gemma4:e4b"


def label_cluster(titles: list[str], fallback: str = "Unlabeled") -> str:
    bullets = "\n".join(f"- {t}" for t in titles)
    prompt = (
        "You are naming a topic cluster. Reply with a 2-4 word topic name only, "
        "no punctuation, no quotes, no explanation.\n\n"
        f"Documents in this cluster:\n{bullets}\n\n"
        "Topic name:"
    )
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "think": False,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 32},
            },
            timeout=60.0,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        first = text.splitlines()[0].strip().strip('"').strip("'.,:")
        if not first or len(first) > 60:
            return fallback
        return first
    except Exception:
        return fallback
