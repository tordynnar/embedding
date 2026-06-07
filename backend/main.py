from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clustering import ClusterCache
from data import DOCS
from embeddings import embed, embed_batch

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    texts = [f"{d['title']}. {d['text']}" for d in DOCS]
    state["doc_embeddings"] = await embed_batch(texts)
    state["cluster_cache"] = ClusterCache()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ClassifyRequest(BaseModel):
    text: str
    k: int = 5


@app.get("/api/docs")
def list_docs():
    return {"docs": DOCS}


@app.post("/api/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    q = await embed(req.query)
    sims = state["doc_embeddings"] @ q
    order = np.argsort(sims)[::-1][: req.top_k]
    results = [
        {**DOCS[i], "score": round(float(sims[i]), 4)}
        for i in order
    ]
    return {"query": req.query, "results": results}


@app.get("/api/clusters")
async def clusters(k: int = 5):
    if k < 2 or k > len(DOCS) - 1:
        raise HTTPException(400, "k out of range")
    result = await state["cluster_cache"].get(state["doc_embeddings"], DOCS, k)
    return {"k": k, "clusters": result["clusters"]}


@app.post("/api/classify")
async def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if req.k < 2 or req.k > len(DOCS) - 1:
        raise HTTPException(400, "k out of range")
    result = await state["cluster_cache"].get(state["doc_embeddings"], DOCS, req.k)
    centroids = result["centroids"]
    v = await embed(req.text)
    centroid_norms = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sims = centroid_norms @ v
    best = int(np.argmax(sims))
    distances = [
        {
            "cluster_id": c["cluster_id"],
            "llm_label": c["llm_label"],
            "similarity": round(float(sims[c["cluster_id"]]), 4),
        }
        for c in result["clusters"]
    ]
    return {
        "assigned_cluster": result["clusters"][best],
        "similarities": distances,
    }
