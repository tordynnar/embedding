import { useEffect, useState } from "react";

export default function ClusterPage({ docs }) {
  const [k, setK] = useState(5);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState("");
  const [classifyResult, setClassifyResult] = useState(null);
  const [classifying, setClassifying] = useState(false);

  useEffect(() => {
    setLoading(true);
    setData(null);
    fetch(`/api/clusters?k=${k}`)
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      });
  }, [k]);

  async function runClassify(e) {
    e.preventDefault();
    if (!text.trim()) return;
    setClassifying(true);
    const r = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, k }),
    });
    setClassifyResult(await r.json());
    setClassifying(false);
  }

  const docById = Object.fromEntries(docs.map((d) => [d.id, d]));
  const maxSim = classifyResult
    ? Math.max(...classifyResult.similarities.map((s) => s.similarity))
    : 0;

  return (
    <section>
      <p className="hint">
        K-means clusters the {docs.length} document embeddings into k groups.
        Each cluster is named by <code>gemma4:e4b</code> from its document titles.
      </p>
      <div className="k-row">
        <label>k =</label>
        {[3, 5, 7].map((v) => (
          <button
            key={v}
            className={v === k ? "active" : ""}
            onClick={() => setK(v)}
          >
            {v}
          </button>
        ))}
      </div>
      {loading && <p className="hint">Clustering and labelling…</p>}
      <div className="cluster-grid">
        {data?.clusters?.map((c) => (
          <article key={c.cluster_id} className="cluster-card">
            <h3>{c.llm_label}</h3>
            <ul>
              {c.doc_ids.map((id) => (
                <li key={id}>{docById[id]?.title}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>

      <h2 className="section-h">Classify your own text</h2>
      <form onSubmit={runClassify} className="classify-form">
        <textarea
          placeholder="Paste a paragraph and we'll embed it and find its nearest cluster…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
        />
        <button type="submit" disabled={classifying || loading}>
          {classifying ? "…" : "Classify"}
        </button>
      </form>
      {classifyResult && (
        <div className="classify-result">
          <p>
            <strong>Assigned cluster:</strong>{" "}
            <span className="badge">
              {classifyResult.assigned_cluster.llm_label}
            </span>
          </p>
          <div className="sim-bars">
            {classifyResult.similarities.map((s) => (
              <div key={s.cluster_id} className="sim-row">
                <span className="sim-label">{s.llm_label}</span>
                <div className="sim-bar-bg">
                  <div
                    className="sim-bar-fg"
                    style={{ width: `${(s.similarity / maxSim) * 100}%` }}
                  />
                </div>
                <span className="sim-value">{s.similarity.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
