"""Smoke tests against the live FastAPI server on :8000 (which uses live Ollama).

Run the server first:
    .venv/bin/uvicorn main:app --port 8000
Then:
    .venv/bin/pytest -q
"""
import httpx

BASE = "http://localhost:8000"


def test_docs_returns_full_corpus():
    r = httpx.get(f"{BASE}/api/docs", timeout=10.0)
    r.raise_for_status()
    docs = r.json()["docs"]
    assert len(docs) == 24
    assert {d["id"] for d in docs} == set(range(1, 25))


def test_search_space_query_returns_space_doc_first():
    r = httpx.post(f"{BASE}/api/search", json={"query": "rocket launch into orbit", "top_k": 3}, timeout=30.0)
    r.raise_for_status()
    results = r.json()["results"]
    space_ids = {5, 6, 7, 8}
    assert results[0]["id"] in space_ids
    assert results[0]["score"] > 0.5


def test_search_cooking_query_returns_cooking_doc_first():
    r = httpx.post(f"{BASE}/api/search", json={"query": "pasta recipe", "top_k": 3}, timeout=30.0)
    r.raise_for_status()
    results = r.json()["results"]
    cooking_ids = {9, 10, 11, 12}
    assert results[0]["id"] in cooking_ids


def test_clusters_partition_all_docs():
    r = httpx.get(f"{BASE}/api/clusters", params={"k": 5}, timeout=300.0)
    r.raise_for_status()
    clusters = r.json()["clusters"]
    assert len(clusters) == 5
    all_ids = [i for c in clusters for i in c["doc_ids"]]
    assert sorted(all_ids) == list(range(1, 25))
    for c in clusters:
        assert c["llm_label"]


def test_classify_ai_text_lands_in_ai_cluster():
    r = httpx.post(
        f"{BASE}/api/classify",
        json={"text": "Neural networks and transformers for language modeling", "k": 5},
        timeout=120.0,
    )
    r.raise_for_status()
    cluster = r.json()["assigned_cluster"]
    ai_ids = {1, 2, 3, 4}
    assert any(i in ai_ids for i in cluster["doc_ids"])
