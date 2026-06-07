import numpy as np
from sklearn.cluster import KMeans

from llm import label_cluster


def cluster(
    doc_embeddings: np.ndarray,
    docs: list[dict],
    k: int,
) -> dict:
    """Return {'centroids': (k,d) ndarray, 'clusters': [{cluster_id, llm_label, doc_ids}]}."""
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    assignments = km.fit_predict(doc_embeddings)
    clusters = []
    for cid in range(k):
        member_titles = [docs[i]["title"] for i, c in enumerate(assignments) if c == cid]
        member_ids = [docs[i]["id"] for i, c in enumerate(assignments) if c == cid]
        llm_name = label_cluster(member_titles, fallback=f"Cluster {cid}")
        clusters.append(
            {
                "cluster_id": cid,
                "llm_label": llm_name,
                "doc_ids": member_ids,
            }
        )
    return {"centroids": km.cluster_centers_, "clusters": clusters}


class ClusterCache:
    def __init__(self):
        self._cache: dict[int, dict] = {}

    def get(self, doc_embeddings: np.ndarray, docs: list[dict], k: int) -> dict:
        if k not in self._cache:
            self._cache[k] = cluster(doc_embeddings, docs, k)
        return self._cache[k]
