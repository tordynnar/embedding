import { useState } from "react";

export default function SearchPage({ docs }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  async function runSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    const r = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    const d = await r.json();
    setResults(d.results);
    setLoading(false);
  }

  const list = results ?? docs;

  return (
    <section>
      <p className="hint">
        Type a natural-language query. The query is embedded, then matched
        against each document's embedding by cosine similarity.
      </p>
      <form onSubmit={runSearch}>
        <input
          type="text"
          placeholder="e.g. machine learning, pasta recipe, deep space telescope…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>
      <div className="result-list">
        {results === null && (
          <p className="hint">
            All {docs.length} documents in the corpus (showing first {docs.length}):
          </p>
        )}
        {list.map((d) => (
          <article key={d.id} className="doc-row">
            <div className="doc-head">
              <span className="doc-title">{d.title}</span>
              {d.score !== undefined && (
                <span className="score">{d.score.toFixed(3)}</span>
              )}
            </div>
            <p className="doc-text">{d.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
