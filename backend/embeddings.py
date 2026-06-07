import numpy as np
from openai import AsyncOpenAI

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:v1.5"

_client = AsyncOpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")


async def embed(text: str) -> np.ndarray:
    return (await embed_batch([text]))[0]


async def embed_batch(texts: list[str]) -> np.ndarray:
    r = await _client.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = np.array([d.embedding for d in r.data], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
