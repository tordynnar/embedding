import { useEffect, useState } from "react";
import SearchPage from "./SearchPage.jsx";
import ClusterPage from "./ClusterPage.jsx";

export default function App() {
  const [tab, setTab] = useState("search");
  const [docs, setDocs] = useState([]);

  useEffect(() => {
    fetch("/api/docs")
      .then((r) => r.json())
      .then((d) => setDocs(d.docs));
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Embedding demos</h1>
        <p className="sub">
          {docs.length} sample documents · <code>nomic-embed-text:v1.5</code> ·{" "}
          <code>gemma4:e4b</code> for cluster labels
        </p>
        <nav>
          <button
            className={tab === "search" ? "active" : ""}
            onClick={() => setTab("search")}
          >
            1. Semantic search
          </button>
          <button
            className={tab === "topics" ? "active" : ""}
            onClick={() => setTab("topics")}
          >
            2. Topic clustering
          </button>
        </nav>
      </header>
      <main>
        {tab === "search" ? (
          <SearchPage docs={docs} />
        ) : (
          <ClusterPage docs={docs} />
        )}
      </main>
    </div>
  );
}
