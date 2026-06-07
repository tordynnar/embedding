import httpx
import numpy as np

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:v1.5"


def embed(text: str) -> np.ndarray:
    r = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30.0,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_batch(texts: list[str]) -> np.ndarray:
    return np.stack([embed(t) for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
