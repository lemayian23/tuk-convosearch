import { useState, useEffect, useCallback } from "react";

// API base. Empty string means "same origin", which is correct in production:
// FastAPI serves this bundle from /admin (see _mount_static in app/main.py), so
// the panel and the API share a host and no CORS is involved. For local dev,
// admin/.env.development sets VITE_API_BASE=http://localhost:8000 while Vite
// runs on :5173.
//   VITE_API_BASE=http://127.0.0.1:8000 npm --prefix admin run build
// builds a bundle pointed somewhere else, but the shipped compose stack needs no
// separate `web`/`app` origin, so leaving it unset is the normal case.
const API = import.meta.env.VITE_API_BASE ?? "";

const TOKEN_KEY = "tuk_admin_token";

// ── helpers ────────────────────────────────────────────────────────
function timeAgo(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

function token() {
  return sessionStorage.getItem(TOKEN_KEY);
}

// 401 anywhere means the JWT expired (TUK_ACCESS_TOKEN_EXPIRE_MINUTES, 8h by
// default) or TUK_SECRET_KEY changed. Drop the token and let App re-render the
// login screen — no full page reload, so whatever the admin was reading is not
// lost on the next click.
let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn;
}

async function readError(res) {
  // FastAPI errors are {detail: "..."}; a proxy error (Caddy 413) may be plain
  // text or empty, so never assume .json() succeeds.
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      // Pydantic validation errors arrive as a list of objects.
      return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return res.statusText || `Request failed (${res.status})`;
  }
}

async function apiFetch(path, opts = {}) {
  const headers = { Accept: "application/json", ...(opts.headers || {}) };
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token()) headers["Authorization"] = `Bearer ${token()}`;
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401) {
    sessionStorage.removeItem(TOKEN_KEY);
    onUnauthorized();
    throw new Error("Your session expired. Sign in again.");
  }
  return res;
}

async function apiJSON(path, opts = {}) {
  const res = await apiFetch(path, opts);
  if (res.status === 204) return null; // DELETE /api/admin/documents/{id}
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

// ── Login page ─────────────────────────────────────────────────────
function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      // POST /api/auth/login -> {access_token, token_type, expires_in_minutes, user}
      const res = await fetch(`${API}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || "Login failed");
      } else {
        sessionStorage.setItem(TOKEN_KEY, data.access_token);
        onLogin(data);
      }
    } catch {
      setError(
        `Cannot reach the API at ${API || window.location.origin}. Is the server running?`
      );
    }
    setLoading(false);
  }

  return (
    <div style={styles.loginWrap}>
      <div style={styles.loginBox}>
        <div style={styles.loginLogo}>🎓</div>
        <h1 style={styles.loginTitle}>TUK-ConvoSearch</h1>
        <p style={styles.loginSub}>Admin Panel</p>
        {error && <div style={styles.alert}>{error}</div>}
        <form onSubmit={handleSubmit}>
          <label style={styles.label}>Email</label>
          <input
            style={styles.input}
            type="email"
            placeholder="you@tuk.ac.ke"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoFocus
          />
          <label style={styles.label}>Password</label>
          <input
            style={styles.input}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button style={styles.btn} type="submit" disabled={loading}>
            {loading ? "Signing in…" : "Sign In"}
          </button>
        </form>
        <p style={{ marginTop: 16, fontSize: 12, color: "#888", textAlign: "center" }}>
          No account yet? On the server:
          <br />
          <code style={{ fontSize: 11 }}>
            docker compose exec app python scripts/create_admin.py
          </code>
        </p>
      </div>
    </div>
  );
}

// ── Documents tab ──────────────────────────────────────────────────
function DocumentsTab({ notify }) {
  const [docs, setDocs] = useState([]);
  const [showInactive, setShowInactive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState(null); // {type: 'success'|'error', text}

  const loadDocs = useCallback(async (includeInactive) => {
    setLoading(true);
    try {
      // GET /api/admin/documents?include_inactive -> [ {document_id, title,
      // filename, file_type, upload_date, chunk_count, is_active, indexed_at} ]
      const data = await apiJSON(
        `/api/admin/documents?include_inactive=${includeInactive ? "true" : "false"}`
      );
      setDocs(data || []);
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.")
        setMsg({ type: "error", text: err.message });
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadDocs(showInactive);
  }, [showInactive, loadDocs]);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setMsg(null);

    // The endpoint takes exactly one multipart field: "file". There is no
    // "title" field any more — the title is derived from the document itself
    // during extraction (app/ingestion/documents.py).
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await apiFetch("/api/admin/documents", { method: "POST", body: form });
      if (res.ok) {
        // 201 {status: indexed|unchanged, filename, document_id, chunks}
        const data = await res.json();
        setMsg({
          type: "success",
          text:
            data.status === "unchanged"
              ? `"${data.filename}" is already indexed and unchanged — nothing was re-embedded.`
              : `Indexed "${data.filename}" (${data.chunks} chunks, document #${data.document_id}).`,
        });
        setFile(null);
        e.target.reset();
        loadDocs(showInactive);
        notify("Document indexed");
      } else {
        // 422 = extracted, but no readable text (scanned PDF). This is the one
        // an admin hits most often, so it gets its own explanation.
        const detail = await readError(res);
        setMsg({
          type: "error",
          text:
            res.status === 422
              ? `No readable text found in "${file.name}". ${detail}\n\nThis usually means a scanned PDF (a photo of pages). Run it through OCR first — e.g. "sudo apt install ocrmypdf; ocrmypdf in.pdf out.pdf" — then upload out.pdf. A .docx saved as an image-only document gives the same result.`
              : res.status === 413
                ? `Too large: ${detail}`
                : res.status === 415
                  ? detail
                  : detail,
        });
      }
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.")
        setMsg({ type: "error", text: err.message });
    }
    setUploading(false);
  }

  async function handleIngestAll() {
    setReindexing(true);
    setMsg(null);
    try {
      // 202 {started, corpus_dir}. It runs in the background: the request
      // returns immediately, so progress is watched on the Index tab.
      const data = await apiJSON("/api/admin/documents/ingest-all", { method: "POST" });
      setMsg({
        type: "success",
        text: `Re-indexing every file in ${data.corpus_dir} in the background. Embedding on CPU takes roughly 1 second per chunk — open the Index tab to watch the chunk count grow.`,
      });
      notify("Corpus re-index started");
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.")
        setMsg({ type: "error", text: err.message });
    }
    setReindexing(false);
  }

  async function handleDelete(doc) {
    if (
      !window.confirm(
        `Remove "${doc.filename}"?\n\nThe document is deactivated and its vectors leave the search index. The file itself is also deleted from the corpus, otherwise the next "Re-index corpus" would bring it back.`
      )
    )
      return;
    try {
      await apiFetch(`/api/admin/documents/${doc.document_id}`, { method: "DELETE" });
      setMsg({ type: "success", text: `Removed "${doc.filename}".` });
      loadDocs(showInactive);
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.")
        setMsg({ type: "error", text: err.message });
    }
  }

  const activeCount = docs.filter((d) => d.is_active).length;

  return (
    <div>
      <h2 style={styles.sectionTitle}>Document Library</h2>

      {/* Upload form */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Upload New Document</h3>
        <form onSubmit={handleUpload}>
          <label style={styles.label}>File (PDF, DOCX, TXT or MD)</label>
          <input
            style={{ ...styles.input, padding: "8px" }}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            required
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button style={styles.btn} type="submit" disabled={uploading || !file}>
            {uploading ? "⏳ Uploading & Indexing…" : "📤 Upload and Index"}
          </button>
        </form>
        {uploading && (
          <p style={{ color: "#888", fontSize: 13, marginTop: 8 }}>
            Embedding runs inside the request (roughly 1 second per chunk on CPU),
            so a long PDF can take 30–90 seconds. Keep this tab open.
          </p>
        )}
        {msg && (
          <div style={msg.type === "success" ? styles.success : styles.alert}>
            <span style={{ whiteSpace: "pre-line" }}>{msg.text}</span>
          </div>
        )}
      </div>

      {/* Bulk re-index */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Re-index the Whole Corpus</h3>
        <p style={{ fontSize: 13, color: "#555", marginTop: 0 }}>
          Reads every supported file in the corpus directory on the server and
          indexes the ones that changed. Use it after you rsync new documents
          straight into <code>corpus/</code> instead of uploading them here —
          unchanged files are skipped by content hash, so it is cheap.
        </p>
        <button
          style={{ ...styles.btn, marginTop: 8, width: "auto", padding: "10px 18px" }}
          onClick={handleIngestAll}
          disabled={reindexing}
        >
          {reindexing ? "⏳ Starting…" : "🔄 Re-index corpus"}
        </button>
      </div>

      {/* Document list */}
      <div style={styles.card}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ ...styles.cardTitle, marginBottom: 0 }}>
            Indexed Documents ({activeCount} active)
          </h3>
          <label style={{ fontSize: 12, color: "#666", display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showInactive}
              onChange={(e) => setShowInactive(e.target.checked)}
            />
            show removed documents
          </label>
        </div>
        {loading ? (
          <p style={{ color: "#888" }}>Loading…</p>
        ) : docs.length === 0 ? (
          <p style={{ color: "#888" }}>
            No documents yet. Upload one above, or re-index the corpus.
          </p>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                {["Filename", "Chunks", "Uploaded", "Indexed", "Status", "Action"].map((h) => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.document_id} style={{ opacity: doc.is_active ? 1 : 0.45 }}>
                  <td style={styles.td}>
                    <span style={{ fontSize: 14 }}>
                      {doc.file_type === "pdf" || doc.filename?.endsWith(".pdf")
                        ? "📄"
                        : doc.file_type === "docx" || doc.filename?.endsWith(".docx")
                          ? "📝"
                          : "📃"}
                    </span>{" "}
                    {doc.title || doc.filename}
                    <br />
                    <span style={{ fontSize: 11, color: "#888" }}>
                      {doc.filename} · id {doc.document_id}
                    </span>
                  </td>
                  <td style={{ ...styles.td, textAlign: "center" }}>{doc.chunk_count}</td>
                  <td style={styles.td}>{timeAgo(doc.upload_date)}</td>
                  <td style={styles.td}>{timeAgo(doc.indexed_at)}</td>
                  <td style={{ ...styles.td, textAlign: "center" }}>
                    <span style={doc.is_active ? styles.badgeGreen : styles.badgeGrey}>
                      {doc.is_active ? "Active" : "Removed"}
                    </span>
                  </td>
                  <td style={{ ...styles.td, textAlign: "center" }}>
                    {doc.is_active ? (
                      <button style={styles.btnDanger} onClick={() => handleDelete(doc)}>
                        Remove
                      </button>
                    ) : (
                      <span style={{ fontSize: 11, color: "#aaa" }}>—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── Index tab (server health: corpus + Ollama) ─────────────────────
function IndexTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      // GET /api/admin/index ->
      // {index:{total_chunks,dimension,embedding_model,index_type,vectors_stored_in},
      //  corpus_dir, corpus_files:[...], documents, ollama:{reachable,models,
      //  expected_model,model_present,error}}
      setData(await apiJSON("/api/admin/index"));
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.") setError(err.message);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) return <p style={{ color: "#888", padding: 24 }}>Loading…</p>;
  if (error) return <div style={styles.alert}>Could not read the index status: {error}</div>;
  if (!data) return null;

  const { index, ollama } = data;
  const rows = [
    ["Chunks in the index", index.total_chunks],
    ["Embedding model", index.embedding_model],
    ["Vector dimension", index.dimension],
    ["Index type", index.index_type],
    ["Vectors stored in", index.vectors_stored_in],
    ["Corpus directory", data.corpus_dir],
    ["Files in corpus", data.corpus_files.length],
    ["Documents in database", data.documents],
  ];

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={styles.sectionTitle}>Index &amp; Model Status</h2>
        <button
          style={{ ...styles.btnDanger, borderColor: "#888", color: "#555", padding: "6px 14px" }}
          onClick={load}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Ollama is the thing that silently breaks: /api/health goes degraded
          when the model is not present, and answers stop without an error page. */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Generation model (Ollama)</h3>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
          <span style={ollama.reachable ? styles.badgeGreen : styles.badgeRed}>
            {ollama.reachable ? "Ollama reachable" : "Ollama unreachable"}
          </span>
          <span style={ollama.model_present ? styles.badgeGreen : styles.badgeRed}>
            {ollama.model_present ? `model present: ${ollama.expected_model}` : `MISSING: ${ollama.expected_model}`}
          </span>
        </div>
        {ollama.error && (
          <div style={styles.alert}>
            {ollama.error}
            <div style={{ fontSize: 12, marginTop: 6 }}>
              If this says the host refused connection, check{" "}
              <code>TUK_OLLAMA_HOST=http://ollama:11434</code> in .env — inside the
              container <code>localhost</code> is the app itself.
            </div>
          </div>
        )}
        {!ollama.model_present && !ollama.error && (
          <div style={styles.alert}>
            The configured model is not installed. Run{" "}
            <code>./deploy/pull-models.sh</code> on the server, then refresh.
          </div>
        )}
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Installed models</th>
              <th style={styles.th}>Configured</th>
            </tr>
          </thead>
          <tbody>
            {(ollama.models || []).map((m, i) => (
              <tr key={m}>
                <td style={styles.td}>
                  {m}
                  {i === 0 && m === ollama.expected_model ? (
                    <span style={{ ...styles.badgeGreen, marginLeft: 8 }}>in use</span>
                  ) : null}
                </td>
                <td style={styles.td}>{m === ollama.expected_model ? "TUK_OLLAMA_MODEL" : ""}</td>
              </tr>
            ))}
            {(ollama.models || []).length === 0 && (
              <tr>
                <td style={styles.td} colSpan={2}>None</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Retrieval index</h3>
        <table style={styles.table}>
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k}>
                <td style={{ ...styles.td, fontWeight: 600, width: 220 }}>{k}</td>
                <td style={styles.td}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Corpus files on disk</h3>
        {data.corpus_files.length === 0 ? (
          <p style={{ color: "#888" }}>
            Nothing in the corpus directory. Upload a document or rsync files into it
            and press "Re-index corpus".
          </p>
        ) : (
          <div>
            {data.corpus_files.map((f) => (
              <span key={f} style={{ ...styles.sourceChip, marginLeft: 0, marginRight: 6 }}>
                {f}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Query Logs tab ─────────────────────────────────────────────────
function LogsTab() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(null);

  async function load() {
    setLoading(true);
    setError("");
    try {
      // Both endpoints answer raw JSON: /logs is a list of rows with `sources`
      // already parsed; /stats is the aggregate object.
      const [l, s] = await Promise.all([
        apiJSON("/api/admin/logs?limit=100"),
        apiJSON("/api/admin/stats"),
      ]);
      setLogs(l || []);
      setStats(s || null);
    } catch (err) {
      if (err.message !== "Your session expired. Sign in again.") setError(err.message);
    }
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={styles.sectionTitle}>Query Logs</h2>
        <button
          style={{ ...styles.btnDanger, borderColor: "#888", color: "#555", padding: "6px 14px" }}
          onClick={load}
        >
          ↻ Refresh
        </button>
      </div>

      {error && <div style={styles.alert}>{error}</div>}

      {stats && (
        <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
          {[
            { label: "Total Queries", value: stats.total_queries },
            { label: "Avg Response Time", value: `${stats.average_response_time}s` },
            { label: "Unanswered", value: stats.unanswered_queries },
            { label: "Ungrounded", value: stats.ungrounded_answers },
          ].map((s) => (
            <div key={s.label} style={styles.statCard}>
              <div style={styles.statValue}>{s.value}</div>
              <div style={styles.statLabel}>{s.label}</div>
            </div>
          ))}
        </div>
      )}

      <div style={styles.card}>
        {loading ? (
          <p style={{ color: "#888" }}>Loading logs…</p>
        ) : logs.length === 0 ? (
          <p style={{ color: "#888" }}>No queries logged yet. Students need to use the chat first.</p>
        ) : (
          logs.map((log) => (
            <div
              key={log.query_id}
              style={styles.logItem}
              onClick={() => setExpanded(expanded === log.query_id ? null : log.query_id)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ fontWeight: 600, fontSize: 14, flex: 1 }}>
                  {log.chunks_found === 0 ? "❌" : log.grounded ? "✅" : "⚠️"} {log.question}
                </div>
                <div style={{ fontSize: 11, color: "#888", marginLeft: 16, whiteSpace: "nowrap" }}>
                  {Number(log.response_time).toFixed(1)}s · {timeAgo(log.timestamp)}
                </div>
              </div>
              {expanded === log.query_id && (
                <div style={styles.logExpanded}>
                  <strong>Answer:</strong>
                  <p style={{ marginTop: 4, color: "#333", fontSize: 13, whiteSpace: "pre-line" }}>
                    {log.answer}
                  </p>
                  {log.sources && log.sources.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <strong>Sources:</strong>
                      {log.sources.map((s, i) => (
                        <span key={i} style={styles.sourceChip} title={s.quote || ""}>
                          {s.source}
                        </span>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: 8, fontSize: 11, color: "#aaa" }}>
                    Route: {log.route || "—"} · Grounded: {log.grounded ? "yes" : "no"} ·
                    Retries: {log.retry_count} · Chunks found: {log.chunks_found} ·
                    Model: {log.model || "—"} · Session: {log.session_id}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ── Stats tab ──────────────────────────────────────────────────────
function StatsTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    // GET /api/admin/stats -> {total_queries, average_response_time,
    // unanswered_queries, ungrounded_answers, top_questions[], index{...},
    // documents, sessions}
    apiJSON("/api/admin/stats")
      .then(setData)
      .catch((err) => {
        if (err.message !== "Your session expired. Sign in again.") setError(err.message);
      });
  }, []);

  if (error) return <div style={styles.alert}>Could not load statistics: {error}</div>;
  if (!data) return <p style={{ color: "#888", padding: 24 }}>Loading…</p>;

  return (
    <div>
      <h2 style={styles.sectionTitle}>System Statistics</h2>

      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        {[
          { label: "Active Documents", value: data.documents },
          { label: "Chunks Indexed", value: data.index?.total_chunks },
          { label: "Total Queries", value: data.total_queries },
          { label: "Unanswered Queries", value: data.unanswered_queries },
          { label: "Ungrounded Answers", value: data.ungrounded_answers },
          { label: "Avg Response Time", value: `${data.average_response_time}s` },
          { label: "Conversations", value: data.sessions },
        ].map((s) => (
          <div key={s.label} style={styles.statCard}>
            <div style={styles.statValue}>{s.value ?? "—"}</div>
            <div style={styles.statLabel}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Unanswered (chunks_found = 0) is the backlog list: every one of these
          is a question the corpus cannot answer yet. */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Most asked questions</h3>
        {(data.top_questions || []).length === 0 ? (
          <p style={{ color: "#888" }}>Nothing yet — students have not asked anything.</p>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Question</th>
                <th style={{ ...styles.th, textAlign: "right" }}>Times asked</th>
              </tr>
            </thead>
            <tbody>
              {data.top_questions.map((q) => (
                <tr key={q.question}>
                  <td style={styles.td}>{q.question}</td>
                  <td style={{ ...styles.td, textAlign: "right" }}>{q.hits}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Index configuration</h3>
        <table style={styles.table}>
          <tbody>
            {Object.entries(data.index || {}).map(([k, v]) => (
              <tr key={k}>
                <td style={{ ...styles.td, fontWeight: 600, width: 200 }}>
                  {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </td>
                <td style={styles.td}>{String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Root App ───────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(null);
  // True while we are validating a stored token. Without it, a returning admin
  // sees the login form flash before /api/auth/me answers.
  const [booting, setBooting] = useState(Boolean(token()));
  const [tab, setTab] = useState("documents");
  const [notice, setNotice] = useState("");

  const notify = useCallback((text) => {
    setNotice(text);
    setTimeout(() => setNotice(""), 4000);
  }, []);

  // Report an expired/invalid token back to the login screen.
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
  }, []);

  // On mount (and after login) confirm the token against GET /api/auth/me,
  // which returns the JWT payload {sub, email, role, iat, exp}. A token from an
  // older deploy (before TUK_SECRET_KEY was rotated) fails here and we land back
  // on the login form instead of showing a page full of 401 errors.
  useEffect(() => {
    if (!token()) {
      setBooting(false);
      return;
    }
    apiJSON("/api/auth/me")
      .then((payload) => setUser({ email: payload.email, role: payload.role }))
      .catch(() => setUser(null))
      .finally(() => setBooting(false));
  }, []);

  function logout() {
    sessionStorage.removeItem(TOKEN_KEY);
    setUser(null);
  }

  if (booting) {
    return (
      <div style={styles.loginWrap}>
        <p style={{ color: "#888", fontSize: 14 }}>Checking your session…</p>
      </div>
    );
  }

  if (!token() || !user) return <LoginPage onLogin={(d) => setUser(d.user)} />;

  const TABS = [
    { id: "documents", icon: "📄", label: "Documents" },
    { id: "index", icon: "🧭", label: "Index & Model" },
    { id: "logs", icon: "📋", label: "Query Logs" },
    { id: "stats", icon: "📊", label: "Statistics" },
  ];

  return (
    <div style={styles.shell}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        <div style={styles.sidebarTop}>
          <div style={styles.sidebarLogo}>🎓</div>
          <div style={styles.sidebarTitle}>TUK-ConvoSearch</div>
          <div style={styles.sidebarSub}>Admin Panel</div>
        </div>
        <nav style={styles.nav}>
          {TABS.map((item) => (
            <button
              key={item.id}
              style={tab === item.id ? styles.navItemActive : styles.navItem}
              onClick={() => setTab(item.id)}
            >
              <span style={{ marginRight: 10 }}>{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <div style={styles.sidebarBottom}>
          <div style={{ ...styles.navItem, cursor: "default", fontSize: 12, color: "#999" }}>
            <span style={{ marginRight: 10 }}>👤</span>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
              {user.email || "signed in"}
              {user.role ? ` · ${user.role}` : ""}
            </span>
          </div>
          {/* Relative to the API base so it works on the VPS and on localhost.
              Under CSP `script-src 'self'` a new tab is fine; nothing inline. */}
          <a href={`${API}/docs`} target="_blank" rel="noreferrer" style={styles.navItem}>
            <span style={{ marginRight: 10 }}>🔗</span>API Docs
          </a>
          <button style={{ ...styles.navItem, color: "#e53935" }} onClick={logout}>
            <span style={{ marginRight: 10 }}>🚪</span>Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div style={styles.main}>
        {notice && <div style={styles.toast}>{notice}</div>}
        {tab === "documents" && <DocumentsTab notify={notify} />}
        {tab === "index" && <IndexTab />}
        {tab === "logs" && <LogsTab />}
        {tab === "stats" && <StatsTab />}
      </div>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────
const styles = {
  // Login
  loginWrap: {
    minHeight: "100vh", display: "flex", alignItems: "center",
    justifyContent: "center", background: "#f7f7f8",
  },
  loginBox: {
    background: "#fff", borderRadius: 16, padding: "40px 36px",
    width: 380, boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
  },
  loginLogo: { fontSize: 40, textAlign: "center", marginBottom: 8 },
  loginTitle: { fontSize: 22, fontWeight: 700, textAlign: "center", color: "#1a1a1a", margin: 0 },
  loginSub: { fontSize: 14, color: "#888", textAlign: "center", marginBottom: 28 },

  // Shell
  shell: { display: "flex", minHeight: "100vh", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" },
  sidebar: {
    width: 240, background: "#1a1a1a", color: "#fff",
    display: "flex", flexDirection: "column", position: "fixed",
    top: 0, left: 0, bottom: 0,
  },
  sidebarTop: { padding: "28px 20px 20px" },
  sidebarLogo: { fontSize: 32, marginBottom: 8 },
  sidebarTitle: { fontSize: 16, fontWeight: 700, color: "#fff" },
  sidebarSub: { fontSize: 12, color: "#888", marginTop: 2 },
  nav: { flex: 1, padding: "8px 12px" },
  navItem: {
    display: "flex", alignItems: "center", width: "100%",
    background: "none", border: "none", color: "#ccc",
    padding: "10px 12px", borderRadius: 8, cursor: "pointer",
    fontSize: 14, textAlign: "left", textDecoration: "none",
    marginBottom: 2, transition: "background 0.15s",
  },
  navItemActive: {
    display: "flex", alignItems: "center", width: "100%",
    background: "#333", border: "none", color: "#fff",
    padding: "10px 12px", borderRadius: 8, cursor: "pointer",
    fontSize: 14, textAlign: "left", marginBottom: 2,
  },
  sidebarBottom: { padding: "12px" },
  main: { marginLeft: 240, flex: 1, padding: "32px", background: "#f7f7f8", minHeight: "100vh" },

  // Components
  sectionTitle: { fontSize: 22, fontWeight: 700, color: "#1a1a1a", marginBottom: 20 },
  card: { background: "#fff", borderRadius: 12, padding: 24, marginBottom: 20, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" },
  cardTitle: { fontSize: 16, fontWeight: 600, color: "#1a1a1a", marginBottom: 16 },
  label: { display: "block", fontSize: 13, fontWeight: 500, color: "#555", marginBottom: 6, marginTop: 12 },
  input: {
    width: "100%", padding: "10px 14px", border: "1px solid #e5e5e5",
    borderRadius: 8, fontSize: 14, outline: "none", boxSizing: "border-box",
    fontFamily: "inherit",
  },
  btn: {
    marginTop: 16, width: "100%", padding: "11px",
    background: "#1a1a1a", color: "#fff", border: "none",
    borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: "pointer",
  },
  btnDanger: {
    padding: "4px 12px", background: "#fff", color: "#e53935",
    border: "1px solid #e53935", borderRadius: 6, fontSize: 12,
    cursor: "pointer",
  },
  alert: {
    marginTop: 12, padding: "10px 14px", background: "#fdecea",
    border: "1px solid #f5c6cb", borderRadius: 8, color: "#c62828", fontSize: 13,
    whiteSpace: "pre-line",
  },
  success: {
    marginTop: 12, padding: "10px 14px", background: "#e8f5e9",
    border: "1px solid #c8e6c9", borderRadius: 8, color: "#2e7d32", fontSize: 13,
  },
  toast: {
    position: "fixed", top: 16, right: 16, zIndex: 10,
    background: "#1a1a1a", color: "#fff", padding: "10px 16px",
    borderRadius: 8, fontSize: 13, boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
  },
  table: { width: "100%", borderCollapse: "collapse" },
  th: {
    textAlign: "left", fontSize: 12, fontWeight: 600,
    color: "#888", padding: "8px 12px", borderBottom: "1px solid #e5e5e5",
    textTransform: "uppercase", letterSpacing: "0.05em",
  },
  td: { padding: "12px 12px", borderBottom: "1px solid #f0f0f0", fontSize: 13, color: "#333" },
  badgeGreen: {
    background: "#e8f5e9", color: "#2e7d32", padding: "2px 10px",
    borderRadius: 20, fontSize: 11, fontWeight: 600,
  },
  badgeGrey: {
    background: "#f0f0f0", color: "#888", padding: "2px 10px",
    borderRadius: 20, fontSize: 11, fontWeight: 600,
  },
  badgeRed: {
    background: "#fdecea", color: "#c62828", padding: "2px 10px",
    borderRadius: 20, fontSize: 11, fontWeight: 600,
  },
  statCard: {
    background: "#fff", borderRadius: 12, padding: "20px 24px",
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)", minWidth: 160,
  },
  statValue: { fontSize: 28, fontWeight: 700, color: "#1a1a1a" },
  statLabel: { fontSize: 13, color: "#888", marginTop: 4 },
  logItem: {
    padding: "14px 0", borderBottom: "1px solid #f0f0f0",
    cursor: "pointer", transition: "background 0.1s",
  },
  logExpanded: {
    marginTop: 12, padding: 14, background: "#f7f7f8",
    borderRadius: 8, fontSize: 13,
  },
  sourceChip: {
    display: "inline-block", background: "#e8f5e9", color: "#2e7d32",
    padding: "2px 10px", borderRadius: 20, fontSize: 11,
    marginLeft: 6, marginTop: 4,
  },
};
