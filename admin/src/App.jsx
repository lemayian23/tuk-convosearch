import { useState, useEffect, useCallback } from "react";

const API = "http://127.0.0.1:8000";

// ── helpers ────────────────────────────────────────────────────────
function timeAgo(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function token() {
  return sessionStorage.getItem("tuk_admin_token");
}

async function apiFetch(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (token()) headers["Authorization"] = `Bearer ${token()}`;
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401) {
    sessionStorage.removeItem("tuk_admin_token");
    window.location.reload();
  }
  return res;
}

// ── Login page ─────────────────────────────────────────────────────
function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("admin@tuk.ac.ke");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Login failed");
      } else {
        sessionStorage.setItem("tuk_admin_token", data.access_token);
        onLogin(data);
      }
    } catch {
      setError("Cannot reach the server. Make sure the backend is running.");
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
          
        </p>
      </div>
    </div>
  );
}

// ── Documents tab ──────────────────────────────────────────────────
function DocumentsTab() {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState(null); // {type: 'success'|'error', text}

  const loadDocs = useCallback(async () => {
    setLoading(true);
    const res = await apiFetch("/api/admin/documents");
    const data = await res.json();
    setDocs(data.documents || []);
    setLoading(false);
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setMsg(null);

    const form = new FormData();
    form.append("file", file);
    form.append("title", title || file.name.replace(/\.[^.]+$/, ""));

    try {
      const res = await fetch(`${API}/api/admin/documents`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token()}` },
        body: form,
      });
      const data = await res.json();
      if (res.ok) {
        setMsg({ type: "success", text: data.message });
        setFile(null);
        setTitle("");
        e.target.reset();
        loadDocs();
      } else {
        setMsg({ type: "error", text: data.detail || "Upload failed" });
      }
    } catch {
      setMsg({ type: "error", text: "Upload request failed." });
    }
    setUploading(false);
  }

  async function handleDelete(doc) {
    if (!window.confirm(`Remove "${doc.filename}" from the search index?\n\nThe file will be kept on disk but will no longer be searchable.`)) return;
    const res = await apiFetch(`/api/admin/documents/${doc.document_id}`, { method: "DELETE" });
    const data = await res.json();
    setMsg({ type: res.ok ? "success" : "error", text: data.message || data.detail });
    loadDocs();
  }

  return (
    <div>
      <h2 style={styles.sectionTitle}>Document Library</h2>

      {/* Upload form */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Upload New Document</h3>
        <form onSubmit={handleUpload}>
          <label style={styles.label}>Document Title (optional)</label>
          <input
            style={styles.input}
            type="text"
            placeholder="e.g. Academic Calendar 2025/2026"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <label style={styles.label}>File (PDF, DOCX, or TXT)</label>
          <input
            style={{ ...styles.input, padding: "8px" }}
            type="file"
            accept=".pdf,.docx,.txt"
            required
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button style={styles.btn} type="submit" disabled={uploading}>
            {uploading ? "⏳ Uploading & Indexing…" : "📤 Upload and Index"}
          </button>
        </form>
        {uploading && (
          <p style={{ color: "#888", fontSize: 13, marginTop: 8 }}>
            This may take 30–90 seconds while the document is embedded into FAISS…
          </p>
        )}
        {msg && (
          <div style={msg.type === "success" ? styles.success : styles.alert}>
            {msg.text}
          </div>
        )}
      </div>

      {/* Document list */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>
          Indexed Documents ({docs.filter((d) => d.is_active).length} active)
        </h3>
        {loading ? (
          <p style={{ color: "#888" }}>Loading…</p>
        ) : docs.length === 0 ? (
          <p style={{ color: "#888" }}>No documents yet. Upload one above.</p>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                {["Filename", "Chunks", "Uploaded", "Status", "Action"].map((h) => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.document_id} style={{ opacity: doc.is_active ? 1 : 0.45 }}>
                  <td style={styles.td}>
                    <span style={{ fontSize: 14 }}>
                      {doc.filename.endsWith(".pdf") ? "📄" : doc.filename.endsWith(".docx") ? "📝" : "📃"}
                    </span>{" "}
                    {doc.title || doc.filename}
                    <br />
                    <span style={{ fontSize: 11, color: "#888" }}>{doc.filename}</span>
                  </td>
                  <td style={{ ...styles.td, textAlign: "center" }}>{doc.chunk_count}</td>
                  <td style={styles.td}>{timeAgo(doc.upload_date)}</td>
                  <td style={{ ...styles.td, textAlign: "center" }}>
                    <span style={doc.is_active ? styles.badgeGreen : styles.badgeGrey}>
                      {doc.is_active ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td style={{ ...styles.td, textAlign: "center" }}>
                    {doc.is_active && (
                      <button
                        style={styles.btnDanger}
                        onClick={() => handleDelete(doc)}
                      >
                        Remove
                      </button>
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

// ── Query Logs tab ─────────────────────────────────────────────────
function LogsTab() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    apiFetch("/api/admin/logs?limit=100")
      .then((r) => r.json())
      .then((data) => {
        setLogs(data.logs || []);
        setStats(data.stats || null);
        setLoading(false);
      });
  }, []);

  return (
    <div>
      <h2 style={styles.sectionTitle}>Query Logs</h2>

      {stats && (
        <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
          {[
            { label: "Total Queries", value: stats.total_queries },
            { label: "Avg Response Time", value: `${stats.average_response_time}s` },
            { label: "Unanswered", value: stats.unanswered_queries },
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
                  {log.chunks_found === 0 ? "❌" : "✅"} {log.question}
                </div>
                <div style={{ fontSize: 11, color: "#888", marginLeft: 16, whiteSpace: "nowrap" }}>
                  {log.response_time.toFixed(1)}s · {timeAgo(log.timestamp)}
                </div>
              </div>
              {expanded === log.query_id && (
                <div style={styles.logExpanded}>
                  <strong>Answer:</strong>
                  <p style={{ marginTop: 4, color: "#333", fontSize: 13 }}>{log.answer}</p>
                  {log.sources && log.sources.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <strong>Sources:</strong>
                      {log.sources.map((s, i) => (
                        <span key={i} style={styles.sourceChip}>{s.source}</span>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: 8, fontSize: 11, color: "#aaa" }}>
                    Session: {log.session_id} · Chunks found: {log.chunks_found}
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

  useEffect(() => {
    apiFetch("/api/admin/stats")
      .then((r) => r.json())
      .then(setData);
  }, []);

  if (!data) return <p style={{ color: "#888", padding: 24 }}>Loading…</p>;

  return (
    <div>
      <h2 style={styles.sectionTitle}>System Statistics</h2>

      <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        {[
          { label: "Active Documents", value: data.documents.total_active },
          { label: "Total Queries", value: data.queries.total_queries },
          { label: "Unanswered Queries", value: data.queries.unanswered_queries },
          { label: "Avg Response Time", value: `${data.queries.average_response_time}s` },
        ].map((s) => (
          <div key={s.label} style={styles.statCard}>
            <div style={styles.statValue}>{s.value}</div>
            <div style={styles.statLabel}>{s.label}</div>
          </div>
        ))}
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>System Configuration</h3>
        <table style={styles.table}>
          <tbody>
            {Object.entries(data.system).map(([k, v]) => (
              <tr key={k}>
                <td style={{ ...styles.td, fontWeight: 600, width: 200 }}>
                  {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </td>
                <td style={styles.td}>{v}</td>
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
  const [user, setUser] = useState(() => {
    const t = sessionStorage.getItem("tuk_admin_token");
    return t ? { token: t } : null;
  });
  const [tab, setTab] = useState("documents");

  function logout() {
    sessionStorage.removeItem("tuk_admin_token");
    setUser(null);
  }

  if (!user) return <LoginPage onLogin={setUser} />;

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
          {[
            { id: "documents", icon: "📄", label: "Documents" },
            { id: "logs", icon: "📋", label: "Query Logs" },
            { id: "stats", icon: "📊", label: "Statistics" },
          ].map((item) => (
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
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            style={styles.navItem}
          >
            <span style={{ marginRight: 10 }}>🔗</span>API Docs
          </a>
          <button style={{ ...styles.navItem, color: "#e53935" }} onClick={logout}>
            <span style={{ marginRight: 10 }}>🚪</span>Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div style={styles.main}>
        {tab === "documents" && <DocumentsTab />}
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
  },
  success: {
    marginTop: 12, padding: "10px 14px", background: "#e8f5e9",
    border: "1px solid #c8e6c9", borderRadius: 8, color: "#2e7d32", fontSize: 13,
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
