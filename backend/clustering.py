import asyncio

import numpy as np
from sklearn.cluster import KMeans

from llm import label_cluster


async def cluster(
    doc_embeddings: np.ndarray,
    docs: list[dict],
    k: int,
) -> dict:
    """Return {'centroids': (k,d) ndarray, 'clusters': [{cluster_id, llm_label, doc_ids}]}."""
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    assignments = km.fit_predict(doc_embeddings)
    member_titles = [
        [docs[i]["title"] for i, c in enumerate(assignments) if c == cid]
        for cid in range(k)
    ]
    member_ids = [
        [docs[i]["id"] for i, c in enumerate(assignments) if c == cid]
        for cid in range(k)
    ]
    labels = await asyncio.gather(*[
        label_cluster(member_titles[cid], fallback=f"Cluster {cid}")
        for cid in range(k)
    ])
    clusters = [
        {"cluster_id": cid, "llm_label": labels[cid], "doc_ids": member_ids[cid]}
        for cid in range(k)
    ]
    return {"centroids": km.cluster_centers_, "clusters": clusters}


class ClusterCache:
    def __init__(self):
        self._cache: dict[int, dict] = {}

    async def get(self, doc_embeddings: np.ndarray, docs: list[dict], k: int) -> dict:
        if k not in self._cache:
            self._cache[k] = await cluster(doc_embeddings, docs, k)
        return self._cache[k]
