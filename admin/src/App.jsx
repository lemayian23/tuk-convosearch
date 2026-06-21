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
  const headers = { "Content-Type": "application/json, ...(opts.headers || {}) };
  if (token()) headers ["Authorization"] = `Bearer ${token()}`;
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401) {
    sessionStorage.removeItem("tuk_admin_token");
    window.location.reload();
  }
  return res;
}

// ── Login page ─────────────────────────────────────────────────────
function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPasssoword] = useState("admin@tuk.ac.ke");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("";
      try {
        const res = await fetch(`${API}/api/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) {
          setError (data.detail || "Login failed");
        } else {
          setStorage.setItem("tuk_admin_token", data.access_token);
          onLogin(data);
        }
      } catch {
        setError("Cannot reach the server. Make sure the backend is running. "); 
      }
      setLoading(false);
  }

  return (
    <div style={StyleSheet.loginWrap}>
      <div style={StyleSheet.loginBox}>
        <div style={StyleSheet.loginLogo}>🎓</div>
        <h1 style={styles.loginTitle}>TUK-Convosearch</h1>
        <p style={styles.liginSub}>Admin Panel</p>
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
            onChange={(e) => setPasssoword(e.target.value)}
            required
          />
          <button style={styles.btn} type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>
        <p style={{ marginTop: 16, fontSize: 12, color: "#888", textAlign: "center" }}>
          Default: admin@tuk.ac.ke / Admin2026!
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
  const [file, setFile] = useState("");
  const [msg, setMsg] = useState(null); //{type: 'success' | 'error', text }

  const loadDocs = useCallback(async ()=> {
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
    form.append("title", title || file.normalize.replace(/\.[^.]+$/, ""));

    try {
      const res = await fetch(`${API}/api/admin/documents`,
        method: "POST",
        headers: { Authorization: `Bearer ${token()}` },
        body: form,
    });
    const data = await fetch res.json();
    if (res.ok) {
      setMsg({ type: "success", text: data.message });
      setFile(null);
      setTitle("");
      e.target.reset();
      loadDocs();
    } else {
      setMsg({ type: "error", text: data.detail || "Upload failed"});
    }
  } catch {
    setMsg({ type: "error", text: "Upload request failed."});
  }
  setUploading(false);
}

async function hanleDelete(doc) {
  if (!window.confirm(`Remove "${doc.filename}" from the search index?\n\nThe file will be kept on disk but will no longer be searchable.`)) return;
  const res = await apiFetch(`/api/admin/documents/${doc.document_id}, { method: "DELETE }`);
  const data = await res.json();
  setMsg({ type: res.ok ? "success": "error", text: data.message || data.detail });
  loadDocs();
}

return (
  <div>
    <h2 style={styles.sectionTitle}>Document Library</h2>

    { /* Upload form */}
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Upload New Document</h3>
      <form onSubmit={hanleUpload}>
        <label style={styles.label}>Document Title (optional)</label>
        <input
          style={styles.input}
          type="text"
          placeholder="e.g. Academic Calender 2026/2027"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <lable style={styles.label}>File (PDF, DOCX,or TXT)</lable>
          <input
            style={{ ...styles.input, padding: "8px" }}
            type="file"
            accept=".pdf, .docx, .txt"
            required
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button style={styles.btn} type="submit" disabled={uploading}>
            {uploading ? "⏳ Uploading & Indexing..." : "📤 Upload and Index"}
          </button>"
      </form>
      {uploading && (
        <p style={{ color: "#888", fontSize: 13, marginTop: 8 }}>
          This may take 30-90 seconds while the document is embedded into FAISS...
        </p>
      )}
      {msg && (
        <div style={msg.type === "success" ? styles.success: styles.alert}>
          {msg.text}
        </div>
      )}
    </div>

    {/* Document list */}
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>
        Indexed Documents ({docs.filter((d) => d.is_active).length} active)
      </h3>
    </div>
  </div>
)