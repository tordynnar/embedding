# Embedding demos

Two small interactive demos showing what a locally-hosted embedding model can
do, served from a single FastAPI backend with a Vite/React frontend (two tabs):

- **Semantic search** — type a natural-language query and rank a small corpus
  by meaning, not keywords.
- **Topic clustering** — group documents by latent topic without labels, name
  each group with a small LLM, and classify new text into the nearest group.

Both demos use models running locally via [Ollama](https://ollama.com):

| Purpose          | Model                     | Size   |
|------------------|---------------------------|--------|
| Embeddings       | `nomic-embed-text:v1.5`   | 274 MB |
| Cluster labels   | `gemma4:e4b`              | 9.6 GB |

The whole stack is a few hundred lines of Python + JSX. It's built to be small
enough to read end-to-end in one sitting.

---

## How it works

### High-level architecture

```
                                           ┌──────────────────────┐
                                           │  Ollama (localhost)  │
                                           │  :11434              │
  ┌─────────────────┐    fetch /api/*       │  - nomic-embed-text  │
  │ React (Vite)    │ ─────────────────┐    │  - gemma4:e4b        │
  │ :5173           │                  │    └──────────┬───────────┘
  │ - SearchPage    │                  ▼               ▲
  │ - ClusterPage   │   ┌─────────────────────────┐    │ httpx POST
  └─────────────────┘   │   FastAPI (uvicorn)     │────┘
                        │   :8000                 │
                        │ - in-memory doc matrix  │
                        │ - cluster cache per k   │
                        └─────────────────────────┘
```

Vite's dev server proxies `/api/*` to `:8000`, so the React code calls
relative URLs and there's no CORS dance in development. CORS is also enabled
server-side as a fallback.

### What an embedding is, in 90 seconds

`nomic-embed-text:v1.5` turns a string into a fixed-length vector of 768
floating-point numbers. The mapping is learned so that **inputs with similar
meaning land near each other in the 768-dimensional space**. "rocket launch"
and "orbital booster" end up close; "rocket launch" and "pasta recipe" end up
far apart.

"Close" here means **cosine similarity**: the angle between the two vectors.
If you normalize both vectors to length 1 (we do, at the source), cosine
similarity simplifies to a plain dot product, so ranking N documents against a
query is one matrix-vector multiply — `(N, 768) @ (768,) = (N,)`.

### Demo 1 — Semantic search

Code path: `backend/main.py:52` (`/api/search`) →
`backend/embeddings.py:9` (`embed`) → NumPy ranking.

```
Startup
─────────────────────────────────────────────────────────
data.py: 24 sample docs (title + 2-4 sentence body)
         │
         │  for d in DOCS:
         │      embed("{title}. {text}")   ──► Ollama
         ▼
  doc_embeddings: ndarray (24, 768), L2-normalized

Per request
─────────────────────────────────────────────────────────
POST /api/search { query, top_k }
         │
         ▼
   embed(query) ─► q: ndarray (768,)
         │
         ▼
   sims = doc_embeddings @ q       # one dot product per doc
   order = argsort(sims)[::-1]     # rank desc
   return [DOCS[i] + {score} for i in order[:top_k]]
```

Embedding is by far the slow step (the model call to Ollama). The dot-product
ranking against 24 docs is microseconds. At small-to-medium scale this brute
force beats any "vector index" you could build.

You'll notice some drift in results: searching "machine learning research"
returns a Mars rover doc in the top 5. That's the model honestly telling you
those embeddings are nearer than the rest of the corpus, not a bug. With a
tiny 24-doc corpus the gaps between topics are narrow.

### Demo 2 — Topic clustering

Code path: `backend/main.py:67` (`/api/clusters`) →
`backend/clustering.py:6` (`cluster`) → `backend/llm.py:7` (`label_cluster`).

```
GET /api/clusters?k=5

   doc_embeddings (24, 768)
         │
         │  KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict
         ▼
   assignments: [2 2 2 2 3 3 3 3 0 0 0 0 1 1 1 4 4 4 4 1 1 1 1 0]
         │
         │  for each cluster:
         │      titles = docs in that cluster
         │      label_cluster(titles, fallback=f"Cluster {cid}")
         │             │
         │             └─► gemma4:e4b   "2-4 word topic name"
         ▼
   [{cluster_id, llm_label, doc_ids}, …]
```

Why k-means: the embeddings already encode meaning in their geometry, so a
simple distance-based partitioning of the embedding space is enough to
recover topical groupings. Random seed is pinned so results are deterministic
across requests for the same k.

#### Naming the clusters

Cluster centroids and document lists aren't human-readable. We feed each
cluster's document titles to `gemma4:e4b` and ask for a 2-4 word topic name
("Home Cooking Fundamentals", "Space exploration breakthroughs", etc.).

Two non-obvious bits in `backend/llm.py`:

- **`"think": False`** — `gemma4:e4b` is a thinking model. Without this
  flag it spends its entire token budget on internal reasoning, returns an
  empty visible response, and the fallback fires. With thinking off, the
  label is the next 4-or-so tokens.
- **A safety net**: if the call fails or returns junk, the function returns
  `fallback` (the caller passes `f"Cluster {cid}"`). The demo never breaks
  because the LLM is flaky.

#### Classifying new text

Code path: `backend/main.py:75` (`/api/classify`).

```
POST /api/classify { text, k }

   centroids (k, 768)             # from the cached k-means fit
   v = embed(text)                # 768-vector for the new text
   centroids_unit = centroids / ||centroids||
   sims = centroids_unit @ v      # similarity to each centroid
   best = argmax(sims)
   return { assigned_cluster, similarities }
```

The UI draws those similarities as a tiny bar chart so you can see not just
*which* cluster won, but *by how much*. A clear winner means the new text
sits squarely in one topical neighborhood; a tight race means it's between
two topics.

### Caching, intentionally simple

- **Doc embeddings**: computed once at startup in `lifespan` and stored as
  a NumPy array on the FastAPI `state` dict. No disk persistence; restart
  the backend and they get re-embedded (~2–3 s).
- **Cluster fits**: lazily computed and cached per `k` in `ClusterCache`
  (`backend/clustering.py:30`). The first call to `/api/clusters?k=5` takes
  a few seconds (k-means + 5 LLM calls); subsequent calls return instantly.
  This also means `/api/classify` is free after the first `/api/clusters?k=k`
  with the same k — it reuses the centroids.
- **Frontend**: docs are fetched once on mount in `App.jsx` and passed down.

### Trade-offs the simple stack makes

- No persistence — restart loses caches but not data (docs are in `data.py`).
- No auth, no rate limiting, no input size caps. Fine for a localhost demo;
  not for the internet.
- Brute-force search and full k-means refits — perfect at this scale, would
  need pgvector / Qdrant / FAISS and incremental clustering past ~100k docs.
- CORS is wide-open (`*`). The Vite proxy means we don't actually need it in
  development, but it's still left on for ergonomics.

---

## Prerequisites

- macOS or Linux
- Python 3.10+
- Node 18+
- [Ollama](https://ollama.com) with both models pulled:
  ```sh
  ollama serve &
  ollama pull nomic-embed-text:v1.5
  ollama pull gemma4:e4b
  ```

## Run

```sh
# Backend (terminal 1)
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 8000
# Startup embeds the 24 docs (~2-3 s) then begins serving.

# Frontend (terminal 2)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Test

With the backend running:

```sh
cd backend
.venv/bin/pytest -q
```

Five smoke tests:

| Test                                          | What it checks                                     |
|-----------------------------------------------|----------------------------------------------------|
| `test_docs_returns_full_corpus`               | `/api/docs` returns all 24 sample docs              |
| `test_search_space_query_returns_space_doc_first` | "rocket launch into orbit" → a space-topic doc at #1 |
| `test_search_cooking_query_returns_cooking_doc_first` | "pasta recipe" → a cooking-topic doc at #1       |
| `test_clusters_partition_all_docs`            | 5 clusters cover all doc ids exactly once, each labeled |
| `test_classify_ai_text_lands_in_ai_cluster`   | AI-themed text lands in the AI cluster              |

These hit the live Ollama server, so they exercise the real embedding model.

## API reference

| Method | Path              | Body / Query                          | Returns                                       |
|--------|-------------------|---------------------------------------|-----------------------------------------------|
| GET    | `/api/docs`       | —                                     | `{ docs: [...] }` (full corpus)                |
| POST   | `/api/search`     | `{ query: str, top_k: int = 5 }`      | `{ query, results: [{...doc, score}] }`        |
| GET    | `/api/clusters`   | `?k=5`                                | `{ k, clusters: [{cluster_id, llm_label, doc_ids}] }` |
| POST   | `/api/classify`   | `{ text: str, k: int = 5 }`           | `{ assigned_cluster, similarities }`           |

## File map

```
backend/
  data.py           24 sample documents (id, title, text)
  embeddings.py     Ollama embed call + L2 normalization + cosine
  llm.py            gemma4 cluster-labeler with think:false and fallback
  clustering.py     k-means + per-k cache
  main.py           FastAPI app, endpoints, CORS, startup embedding cache
  test_backend.py   Live smoke tests against the running server
  requirements.txt
frontend/
  vite.config.js    /api proxy to :8000
  index.html        root mount point
  src/
    main.jsx        ReactDOM bootstrap
    App.jsx         Tab switcher, loads /api/docs once
    SearchPage.jsx  Query + ranked results with similarity scores
    ClusterPage.jsx Cluster cards + classify form + similarity bars
    styles.css      All styling
```
