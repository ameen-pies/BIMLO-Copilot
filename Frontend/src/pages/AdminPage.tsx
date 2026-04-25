import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Shield, Users, Activity, Trash2, Mail, Edit3, Check, X,
  RefreshCw, LogOut, ChevronDown, Search, Terminal, Zap,
  TrendingUp, MessageSquare, FileText, BarChart2, Clock,
  Crown, User, AlertCircle, Send, Eye, EyeOff, ArrowLeft,
  Wifi, WifiOff, Circle,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Types ───────────────────────────────────────────────────────────────────

interface AdminUser {
  user_id:            string;
  username:           string;
  email:              string;
  role:               string;
  created_at:         string;
  last_seen:          string;
  avatar_url:         string;
  conversation_count: number;
  document_count:     number;
}

interface Stats {
  total_users:         number;
  admin_users:         number;
  active_1h:           number;
  active_24h:          number;
  new_users_7d:        number;
  total_conversations: number;
  total_documents:     number;
  total_reports:       number;
}

interface LogEntry { ts: string; msg: string; }

// ─── Helpers ─────────────────────────────────────────────────────────────────

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)  return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function isOnline(last_seen: string | null | undefined): boolean {
  if (!last_seen) return false;
  return Date.now() - new Date(last_seen).getTime() < 5 * 60 * 1000; // 5 min
}

function classifyLog(msg: string): "error" | "warn" | "success" | "info" {
  const m = msg.toLowerCase();
  if (m.includes("❌") || m.includes("error") || m.includes("failed") || m.includes("exception")) return "error";
  if (m.includes("⚠️") || m.includes("warn") || m.includes("missing")) return "warn";
  if (m.includes("✅") || m.includes("success") || m.includes("ready") || m.includes("connected") || m.includes("saved")) return "success";
  return "info";
}

function authHeaders(token: string) {
  return { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

function KpiCard({ icon: Icon, label, value, sub, accent }: {
  icon: React.ElementType; label: string; value: number | string; sub?: string; accent: string;
}) {
  return (
    <div style={{
      background: "hsl(var(--card))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 16,
      padding: "20px 22px",
      display: "flex", flexDirection: "column", gap: 8,
      position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", top: 0, right: 0, width: 80, height: 80,
        background: `radial-gradient(circle at 70% 30%, ${accent}22, transparent 70%)`,
        pointerEvents: "none",
      }} />
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8, background: `${accent}18`,
          border: `1px solid ${accent}30`, display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Icon size={15} color={accent} />
        </div>
        <span style={{ fontSize: 11, fontWeight: 600, color: "hsl(var(--muted-foreground))", letterSpacing: "0.04em", textTransform: "uppercase" }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "hsl(var(--foreground))", lineHeight: 1.1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }}>{sub}</div>}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function AdminPage() {
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab]           = useState<"users" | "logs" | "email" | "settings">("users");
  const [users, setUsers]       = useState<AdminUser[]>([]);
  const [stats, setStats]       = useState<Stats | null>(null);
  const [logs, setLogs]         = useState<LogEntry[]>([]);
  const [search, setSearch]     = useState("");
  const [loading, setLoading]   = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);

  // Edit modal
  const [editUser, setEditUser]       = useState<AdminUser | null>(null);
  const [editUsername, setEditUsername] = useState("");
  const [editEmail, setEditEmail]       = useState("");
  const [editPassword, setEditPassword] = useState("");
  const [editRole, setEditRole]         = useState<"user" | "admin">("user");
  const [showPw, setShowPw]             = useState(false);
  const [editSaving, setEditSaving]     = useState(false);

  // Delete confirm
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleting, setDeleting]           = useState(false);

  // Email modal
  const [emailTargets, setEmailTargets] = useState<string[]>([]);
  const [emailSubject, setEmailSubject] = useState("");
  const [emailBody, setEmailBody]       = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [emailResult, setEmailResult]   = useState<string | null>(null);
  const [showEmailModal, setShowEmailModal] = useState(false);

  // Self-credentials panel
  const [selfUsername, setSelfUsername] = useState(currentUser?.username || "");
  const [selfEmail, setSelfEmail]       = useState(currentUser?.email || "");
  const [selfPassword, setSelfPassword] = useState("");
  const [selfSaving, setSelfSaving]     = useState(false);
  const [selfMsg, setSelfMsg]           = useState<string | null>(null);

  const logEndRef  = useRef<HTMLDivElement>(null);
  const esRef      = useRef<EventSource | null>(null);

  // ── Auth guard ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!currentUser) { navigate("/"); return; }
    if (currentUser.role !== "admin") { navigate("/"); return; }
  }, [currentUser, navigate]);

  // ── Load users + stats ──────────────────────────────────────────────────────
  const loadUsers = useCallback(async () => {
    if (!currentUser?.token) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/auth/admin/users`, { headers: authHeaders(currentUser.token) });
      if (res.ok) setUsers(await res.json());
    } finally { setLoading(false); }
  }, [currentUser?.token]);

  const loadStats = useCallback(async () => {
    if (!currentUser?.token) return;
    setStatsLoading(true);
    try {
      const res = await fetch(`${API}/auth/admin/stats`, { headers: authHeaders(currentUser.token) });
      if (res.ok) setStats(await res.json());
    } finally { setStatsLoading(false); }
  }, [currentUser?.token]);

  useEffect(() => {
    loadUsers();
    loadStats();
  }, [loadUsers, loadStats]);

  // ── Live log SSE ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (tab !== "logs" || !currentUser?.token) return;
    if (esRef.current) { esRef.current.close(); esRef.current = null; }

    // Fetch buffered logs first
    fetch(`${API}/auth/admin/logs?limit=200`, { headers: authHeaders(currentUser.token) })
      .then(r => r.json())
      .then(d => setLogs(d.logs || []))
      .catch(() => {});

    // Then open SSE
    const es = new EventSource(`${API}/auth/admin/logs/stream?token=${currentUser.token}`);
    esRef.current = es;
    es.onmessage = (e) => {
      try {
        const entry: LogEntry = JSON.parse(e.data);
        if (!entry.msg) return;
        setLogs(prev => [...prev.slice(-499), entry]);
      } catch {}
    };
    return () => { es.close(); esRef.current = null; };
  }, [tab, currentUser?.token]);

  useEffect(() => {
    if (tab === "logs") logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, tab]);

  // ── Edit user ───────────────────────────────────────────────────────────────
  function openEdit(u: AdminUser) {
    setEditUser(u); setEditUsername(u.username); setEditEmail(u.email);
    setEditPassword(""); setEditRole(u.role as "user" | "admin"); setShowPw(false);
  }

  async function saveEdit() {
    if (!editUser || !currentUser?.token) return;
    setEditSaving(true);
    const body: any = {};
    if (editUsername !== editUser.username) body.username = editUsername;
    if (editEmail    !== editUser.email)    body.email    = editEmail;
    if (editPassword)                       body.password = editPassword;
    if (editRole     !== editUser.role)     body.role     = editRole;
    try {
      const res = await fetch(`${API}/auth/admin/users/${editUser.user_id}`, {
        method: "PATCH", headers: authHeaders(currentUser.token), body: JSON.stringify(body),
      });
      if (res.ok) { await loadUsers(); setEditUser(null); }
    } finally { setEditSaving(false); }
  }

  // ── Delete user ─────────────────────────────────────────────────────────────
  async function confirmDelete() {
    if (!deleteConfirm || !currentUser?.token) return;
    setDeleting(true);
    try {
      await fetch(`${API}/auth/admin/users/${deleteConfirm}`, {
        method: "DELETE", headers: authHeaders(currentUser.token),
      });
      setUsers(prev => prev.filter(u => u.user_id !== deleteConfirm));
      setDeleteConfirm(null);
      loadStats();
    } finally { setDeleting(false); }
  }

  // ── Send email ──────────────────────────────────────────────────────────────
  async function sendEmail() {
    if (!currentUser?.token || !emailSubject || !emailBody || !emailTargets.length) return;
    setEmailSending(true); setEmailResult(null);
    try {
      const res = await fetch(`${API}/auth/admin/send-email`, {
        method: "POST", headers: authHeaders(currentUser.token),
        body: JSON.stringify({ user_ids: emailTargets, subject: emailSubject, body: emailBody }),
      });
      const d = await res.json();
      setEmailResult(`Sent to ${d.sent?.length || 0} users${d.failed?.length ? `, ${d.failed.length} failed` : ""}.`);
    } finally { setEmailSending(false); }
  }

  // ── Self update ─────────────────────────────────────────────────────────────
  async function saveSelf() {
    if (!currentUser?.token) return;
    setSelfSaving(true); setSelfMsg(null);
    const body: any = {};
    if (selfUsername !== currentUser.username) body.username = selfUsername;
    if (selfEmail    !== currentUser.email)    body.email    = selfEmail;
    if (selfPassword)                          body.password = selfPassword;
    try {
      const res = await fetch(`${API}/auth/admin/users/${currentUser.user_id}`, {
        method: "PATCH", headers: authHeaders(currentUser.token), body: JSON.stringify(body),
      });
      if (res.ok) { setSelfMsg("Credentials updated! Please log in again."); setSelfPassword(""); }
      else        { setSelfMsg("Update failed."); }
    } finally { setSelfSaving(false); }
  }

  const filtered = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  );

  const onlineCount  = users.filter(u => isOnline(u.last_seen)).length;
  const offlineCount = users.length - onlineCount;

  if (!currentUser || currentUser.role !== "admin") return null;

  // ── Styles ──────────────────────────────────────────────────────────────────
  const S = {
    page: {
      minHeight: "100vh",
      background: "hsl(var(--background))",
      fontFamily: "'Geist', 'DM Sans', system-ui, sans-serif",
    } as React.CSSProperties,
    header: {
      position: "sticky" as const, top: 0, zIndex: 50,
      background: "hsl(var(--background)/0.85)",
      backdropFilter: "blur(12px)",
      borderBottom: "1px solid hsl(var(--border)/0.5)",
      padding: "0 28px",
      height: 58,
      display: "flex", alignItems: "center", justifyContent: "space-between",
    },
    tab: (active: boolean) => ({
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "6px 14px", borderRadius: 8, fontSize: 13, fontWeight: 600,
      cursor: "pointer", border: "none",
      background: active ? "hsl(var(--primary))" : "transparent",
      color: active ? "hsl(var(--primary-foreground))" : "hsl(var(--muted-foreground))",
      transition: "all 0.15s",
    }) as React.CSSProperties,
    card: {
      background: "hsl(var(--card))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 16,
    } as React.CSSProperties,
    input: {
      width: "100%", padding: "8px 12px", borderRadius: 8, fontSize: 13,
      background: "hsl(var(--muted)/0.4)",
      border: "1px solid hsl(var(--border))",
      color: "hsl(var(--foreground))",
      outline: "none",
    } as React.CSSProperties,
    btn: (variant: "primary" | "ghost" | "danger" = "primary") => ({
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "7px 14px", borderRadius: 8, fontSize: 12, fontWeight: 600,
      cursor: "pointer", border: "none", transition: "opacity 0.15s",
      background: variant === "primary" ? "hsl(var(--primary))"
                : variant === "danger"  ? "#ef4444"
                : "hsl(var(--muted)/0.6)",
      color: variant === "ghost" ? "hsl(var(--foreground))" : "#fff",
    }) as React.CSSProperties,
  };

  return (
    <div style={S.page}>
      {/* ── Header ── */}
      <div style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <Link to="/" style={{ display: "flex", alignItems: "center", gap: 6, color: "hsl(var(--muted-foreground))", textDecoration: "none", fontSize: 13 }}>
            <ArrowLeft size={14} /> Home
          </Link>
          <div style={{ width: 1, height: 18, background: "hsl(var(--border))" }} />
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8,
              background: "linear-gradient(135deg,#7c3aed,#4f46e5)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Shield size={14} color="#fff" />
            </div>
            <span style={{ fontSize: 15, fontWeight: 800, color: "hsl(var(--foreground))" }}>Admin Dashboard</span>
            <span style={{
              fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
              padding: "2px 7px", borderRadius: 999,
              background: "linear-gradient(135deg,#7c3aed22,#4f46e522)",
              border: "1px solid #7c3aed44", color: "#7c3aed",
            }}>BIMLO</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {/* Online indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "hsl(var(--muted-foreground))" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <Circle size={7} fill="#22c55e" color="#22c55e" />
              <span style={{ color: "#22c55e", fontWeight: 600 }}>{onlineCount}</span>
            </div>
            <span>/</span>
            <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <Circle size={7} fill="#64748b" color="#64748b" />
              <span>{offlineCount}</span>
            </div>
            <span>users</span>
          </div>
          <button onClick={() => { loadUsers(); loadStats(); }} style={S.btn("ghost")}>
            <RefreshCw size={13} />
          </button>
          <button onClick={logout} style={S.btn("ghost")}>
            <LogOut size={13} /> Logout
          </button>
        </div>
      </div>

      <div style={{ padding: "24px 28px", maxWidth: 1280, margin: "0 auto" }}>

        {/* ── KPIs ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 14, marginBottom: 24 }}>
          <KpiCard icon={Users}       label="Total Users"    value={stats?.total_users ?? "—"}         sub={`${stats?.admin_users ?? 0} admin(s)`}    accent="#3b82f6" />
          <KpiCard icon={Wifi}        label="Active (1h)"    value={stats?.active_1h ?? "—"}           sub="online now"                               accent="#22c55e" />
          <KpiCard icon={Activity}    label="Active (24h)"   value={stats?.active_24h ?? "—"}          sub="last 24 hours"                            accent="#06b6d4" />
          <KpiCard icon={TrendingUp}  label="New (7d)"       value={stats?.new_users_7d ?? "—"}        sub="new signups"                              accent="#a855f7" />
          <KpiCard icon={MessageSquare} label="Conversations" value={stats?.total_conversations ?? "—"} sub="all sessions"                            accent="#f59e0b" />
          <KpiCard icon={FileText}    label="Documents"      value={stats?.total_documents ?? "—"}     sub="uploaded"                                 accent="#ec4899" />
          <KpiCard icon={BarChart2}   label="Reports"        value={stats?.total_reports ?? "—"}       sub="generated"                                accent="#14b8a6" />
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20 }}>
          {(["users", "logs", "email", "settings"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} style={S.tab(tab === t)}>
              {t === "users"    && <><Users size={13} /> Users</>}
              {t === "logs"     && <><Terminal size={13} /> Live Logs</>}
              {t === "email"    && <><Mail size={13} /> Send Email</>}
              {t === "settings" && <><Shield size={13} /> My Account</>}
            </button>
          ))}
        </div>

        {/* ══════════════════════ USERS TAB ══════════════════════ */}
        {tab === "users" && (
          <div style={S.card}>
            {/* Toolbar */}
            <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ position: "relative", flex: 1, maxWidth: 320 }}>
                <Search size={13} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "hsl(var(--muted-foreground))" }} />
                <input
                  value={search} onChange={e => setSearch(e.target.value)}
                  placeholder="Search users…"
                  style={{ ...S.input, paddingLeft: 30 }}
                />
              </div>
              <span style={{ fontSize: 12, color: "hsl(var(--muted-foreground))", marginLeft: "auto" }}>
                {filtered.length} user{filtered.length !== 1 ? "s" : ""}
              </span>
            </div>

            {/* Table */}
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid hsl(var(--border))" }}>
                    {["User", "Email", "Role", "Status", "Conversations", "Documents", "Last seen", "Actions"].map(h => (
                      <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", letterSpacing: "0.05em", textTransform: "uppercase", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    <tr><td colSpan={8} style={{ padding: 40, textAlign: "center", color: "hsl(var(--muted-foreground))" }}>
                      <RefreshCw size={18} style={{ animation: "spin 0.8s linear infinite", margin: "0 auto" }} />
                    </td></tr>
                  ) : filtered.length === 0 ? (
                    <tr><td colSpan={8} style={{ padding: 40, textAlign: "center", color: "hsl(var(--muted-foreground))", fontSize: 13 }}>No users found</td></tr>
                  ) : filtered.map(u => {
                    const online = isOnline(u.last_seen);
                    const isMe   = u.user_id === currentUser?.user_id;
                    return (
                      <tr key={u.user_id} style={{
                        borderBottom: "1px solid hsl(var(--border)/0.5)",
                        background: isMe ? "hsl(var(--primary)/0.04)" : "transparent",
                        transition: "background 0.1s",
                      }}
                      onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = isMe ? "hsl(var(--primary)/0.07)" : "hsl(var(--muted)/0.3)"}
                      onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = isMe ? "hsl(var(--primary)/0.04)" : "transparent"}
                      >
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{
                              width: 32, height: 32, borderRadius: "50%",
                              background: u.avatar_url ? "transparent" : (u.role === "admin" ? "linear-gradient(135deg,#7c3aed,#4f46e5)" : "linear-gradient(135deg,#3b82f6,#06b6d4)"),
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: 13, fontWeight: 700, color: "#fff", flexShrink: 0, overflow: "hidden",
                            }}>
                              {u.avatar_url ? <img src={u.avatar_url} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} referrerPolicy="no-referrer" /> : u.username[0]?.toUpperCase()}
                            </div>
                            <div>
                              <div style={{ fontSize: 13, fontWeight: 600, color: "hsl(var(--foreground))" }}>
                                {u.username} {isMe && <span style={{ fontSize: 10, color: "#7c3aed", fontWeight: 700 }}>(you)</span>}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 12, color: "hsl(var(--muted-foreground))" }}>{u.email}</td>
                        <td style={{ padding: "12px 16px" }}>
                          <span style={{
                            fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 999,
                            background: u.role === "admin" ? "#7c3aed22" : "#3b82f622",
                            color: u.role === "admin" ? "#7c3aed" : "#3b82f6",
                            border: `1px solid ${u.role === "admin" ? "#7c3aed44" : "#3b82f644"}`,
                            display: "inline-flex", alignItems: "center", gap: 4,
                          }}>
                            {u.role === "admin" ? <Crown size={9} /> : <User size={9} />}
                            {u.role}
                          </span>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12 }}>
                            <Circle size={7} fill={online ? "#22c55e" : "#64748b"} color={online ? "#22c55e" : "#64748b"} />
                            <span style={{ color: online ? "#22c55e" : "hsl(var(--muted-foreground))" }}>{online ? "Online" : "Offline"}</span>
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 13, color: "hsl(var(--foreground))", textAlign: "center" }}>{u.conversation_count}</td>
                        <td style={{ padding: "12px 16px", fontSize: 13, color: "hsl(var(--foreground))", textAlign: "center" }}>{u.document_count}</td>
                        <td style={{ padding: "12px 16px", fontSize: 12, color: "hsl(var(--muted-foreground))", whiteSpace: "nowrap" }}>{timeAgo(u.last_seen)}</td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <button onClick={() => openEdit(u)} title="Edit" style={{
                              width: 28, height: 28, borderRadius: 6, border: "1px solid hsl(var(--border))",
                              background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                              color: "hsl(var(--muted-foreground))",
                            }}><Edit3 size={12} /></button>
                            <button
                              onClick={() => { setEmailTargets([u.user_id]); setShowEmailModal(true); }}
                              title="Email"
                              style={{ width: 28, height: 28, borderRadius: 6, border: "1px solid hsl(var(--border))", background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "hsl(var(--muted-foreground))" }}
                            ><Mail size={12} /></button>
                            {!isMe && (
                              <button onClick={() => setDeleteConfirm(u.user_id)} title="Delete" style={{
                                width: 28, height: 28, borderRadius: 6, border: "1px solid #ef444430",
                                background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                                color: "#ef4444",
                              }}><Trash2 size={12} /></button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ══════════════════════ LOGS TAB ══════════════════════ */}
        {tab === "logs" && (
          <div style={{ ...S.card, overflow: "hidden" }}>
            <div style={{
              padding: "12px 20px", borderBottom: "1px solid hsl(var(--border))",
              display: "flex", alignItems: "center", gap: 10,
              background: "hsl(var(--muted)/0.2)",
            }}>
              <Terminal size={14} color="#22c55e" />
              <span style={{ fontSize: 13, fontWeight: 700, color: "hsl(var(--foreground))" }}>System Logs</span>
              <div style={{ display: "flex", alignItems: "center", gap: 5, marginLeft: "auto" }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite" }} />
                <span style={{ fontSize: 11, color: "#22c55e", fontWeight: 600 }}>LIVE</span>
              </div>
              <span style={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }}>{logs.length} entries</span>
              <button onClick={() => setLogs([])} style={S.btn("ghost")}><X size={12} /> Clear</button>
            </div>
            <div style={{
              height: 560, overflowY: "auto", padding: "8px 0",
              background: "hsl(220 15% 7%)",
              fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
              fontSize: 12,
            }}>
              {logs.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: "#4a5568", fontSize: 13 }}>
                  Waiting for log entries…
                </div>
              ) : logs.map((l, i) => {
                const kind = classifyLog(l.msg);
                const color = kind === "error" ? "#f87171" : kind === "warn" ? "#fbbf24" : kind === "success" ? "#4ade80" : "#94a3b8";
                const bg    = kind === "error" ? "rgba(248,113,113,0.04)" : kind === "warn" ? "rgba(251,191,36,0.04)" : "transparent";
                const ts = l.ts ? new Date(l.ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
                return (
                  <div key={i} style={{ display: "flex", gap: 12, padding: "3px 16px", background: bg, transition: "background 0.1s" }}
                    onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.03)"}
                    onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = bg}
                  >
                    <span style={{ color: "#4a5568", flexShrink: 0, userSelect: "none", fontSize: 11 }}>{ts}</span>
                    <span style={{ color, wordBreak: "break-all", lineHeight: 1.6 }}>{l.msg}</span>
                  </div>
                );
              })}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* ══════════════════════ EMAIL TAB ══════════════════════ */}
        {tab === "email" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Select recipients */}
            <div style={S.card}>
              <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))" }}>
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>Select Recipients</h3>
              </div>
              <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", gap: 8, marginBottom: 4 }}>
                  <button onClick={() => setEmailTargets(users.map(u => u.user_id))} style={S.btn("ghost")}>All users</button>
                  <button onClick={() => setEmailTargets([])} style={S.btn("ghost")}>Clear</button>
                </div>
                <div style={{ maxHeight: 360, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
                  {users.map(u => {
                    const selected = emailTargets.includes(u.user_id);
                    return (
                      <label key={u.user_id} style={{
                        display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
                        borderRadius: 8, cursor: "pointer",
                        background: selected ? "hsl(var(--primary)/0.08)" : "transparent",
                        border: `1px solid ${selected ? "hsl(var(--primary)/0.3)" : "transparent"}`,
                        transition: "all 0.12s",
                      }}>
                        <input type="checkbox" checked={selected}
                          onChange={() => setEmailTargets(prev => selected ? prev.filter(id => id !== u.user_id) : [...prev, u.user_id])}
                          style={{ accentColor: "hsl(var(--primary))" }}
                        />
                        <div style={{ width: 24, height: 24, borderRadius: "50%", background: "linear-gradient(135deg,#3b82f6,#06b6d4)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "#fff" }}>
                          {u.username[0]?.toUpperCase()}
                        </div>
                        <div>
                          <div style={{ fontSize: 13, fontWeight: 600, color: "hsl(var(--foreground))" }}>{u.username}</div>
                          <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }}>{u.email}</div>
                        </div>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Compose */}
            <div style={S.card}>
              <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))" }}>
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>Compose Message</h3>
              </div>
              <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Subject</label>
                  <input value={emailSubject} onChange={e => setEmailSubject(e.target.value)} placeholder="Email subject…" style={S.input} />
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Body</label>
                  <textarea value={emailBody} onChange={e => setEmailBody(e.target.value)} placeholder="Write your message…" rows={8}
                    style={{ ...S.input, resize: "vertical", fontFamily: "inherit" }} />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <button onClick={sendEmail} disabled={emailSending || !emailTargets.length || !emailSubject || !emailBody}
                    style={{ ...S.btn("primary"), opacity: emailSending || !emailTargets.length || !emailSubject || !emailBody ? 0.5 : 1 }}>
                    {emailSending ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} /> : <Send size={13} />}
                    Send to {emailTargets.length} user{emailTargets.length !== 1 ? "s" : ""}
                  </button>
                  {emailResult && <span style={{ fontSize: 12, color: "#22c55e" }}>{emailResult}</span>}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════ SETTINGS TAB ══════════════════════ */}
        {tab === "settings" && (
          <div style={{ maxWidth: 480 }}>
            <div style={S.card}>
              <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", gap: 8 }}>
                <Shield size={15} color="#7c3aed" />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>My Admin Credentials</h3>
              </div>
              <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Username</label>
                  <input value={selfUsername} onChange={e => setSelfUsername(e.target.value)} style={S.input} />
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Email</label>
                  <input value={selfEmail} onChange={e => setSelfEmail(e.target.value)} type="email" style={S.input} />
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>New Password <span style={{ fontWeight: 400 }}>(leave blank to keep)</span></label>
                  <div style={{ position: "relative" }}>
                    <input value={selfPassword} onChange={e => setSelfPassword(e.target.value)} type={showPw ? "text" : "password"} placeholder="••••••••" style={{ ...S.input, paddingRight: 36 }} />
                    <button onClick={() => setShowPw(v => !v)} style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "hsl(var(--muted-foreground))", padding: 4 }}>
                      {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                </div>
                <button onClick={saveSelf} disabled={selfSaving} style={{ ...S.btn("primary"), width: "fit-content", opacity: selfSaving ? 0.6 : 1 }}>
                  {selfSaving ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} /> : <Check size={13} />}
                  Save Changes
                </button>
                {selfMsg && <div style={{ fontSize: 12, color: selfMsg.includes("fail") ? "#ef4444" : "#22c55e", padding: "8px 12px", borderRadius: 8, background: selfMsg.includes("fail") ? "#ef444415" : "#22c55e15" }}>{selfMsg}</div>}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ══════════ EDIT USER MODAL ══════════ */}
      {editUser && (
        <div style={{ position: "fixed", inset: 0, zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
          onClick={e => { if (e.target === e.currentTarget) setEditUser(null); }}>
          <div style={{ ...S.card, width: 420, padding: 0, overflow: "hidden", boxShadow: "0 24px 80px rgba(0,0,0,0.4)" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>Edit User — {editUser.username}</h3>
              <button onClick={() => setEditUser(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "hsl(var(--muted-foreground))", padding: 4 }}><X size={16} /></button>
            </div>
            <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Username</label>
                <input value={editUsername} onChange={e => setEditUsername(e.target.value)} style={S.input} />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Email</label>
                <input value={editEmail} onChange={e => setEditEmail(e.target.value)} type="email" style={S.input} />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>New Password <span style={{ fontWeight: 400 }}>(leave blank to keep)</span></label>
                <div style={{ position: "relative" }}>
                  <input value={editPassword} onChange={e => setEditPassword(e.target.value)} type={showPw ? "text" : "password"} placeholder="••••••••" style={{ ...S.input, paddingRight: 36 }} />
                  <button onClick={() => setShowPw(v => !v)} style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "hsl(var(--muted-foreground))", padding: 4 }}>
                    {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Role</label>
                <div style={{ display: "flex", gap: 8 }}>
                  {(["user", "admin"] as const).map(r => (
                    <button key={r} onClick={() => setEditRole(r)} style={{
                      flex: 1, padding: "8px 0", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer",
                      border: `1px solid ${editRole === r ? (r === "admin" ? "#7c3aed" : "#3b82f6") : "hsl(var(--border))"}`,
                      background: editRole === r ? (r === "admin" ? "#7c3aed22" : "#3b82f622") : "transparent",
                      color: editRole === r ? (r === "admin" ? "#7c3aed" : "#3b82f6") : "hsl(var(--muted-foreground))",
                      display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 5,
                    }}>
                      {r === "admin" ? <Crown size={12} /> : <User size={12} />} {r}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, paddingTop: 4 }}>
                <button onClick={saveEdit} disabled={editSaving} style={{ ...S.btn("primary"), flex: 1, justifyContent: "center", opacity: editSaving ? 0.6 : 1 }}>
                  {editSaving ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} /> : <Check size={13} />} Save
                </button>
                <button onClick={() => setEditUser(null)} style={{ ...S.btn("ghost"), flex: 1, justifyContent: "center" }}><X size={13} /> Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════ DELETE CONFIRM MODAL ══════════ */}
      {deleteConfirm && (
        <div style={{ position: "fixed", inset: 0, zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
          onClick={e => { if (e.target === e.currentTarget) setDeleteConfirm(null); }}>
          <div style={{ ...S.card, width: 360, padding: 24, boxShadow: "0 24px 80px rgba(0,0,0,0.4)" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, textAlign: "center" }}>
              <div style={{ width: 48, height: 48, borderRadius: "50%", background: "#ef444422", border: "1px solid #ef444440", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <AlertCircle size={22} color="#ef4444" />
              </div>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "hsl(var(--foreground))" }}>Delete User?</h3>
              <p style={{ margin: 0, fontSize: 13, color: "hsl(var(--muted-foreground))" }}>
                This will permanently delete the user and all their conversations, documents, and data. This cannot be undone.
              </p>
              <div style={{ display: "flex", gap: 10, width: "100%", marginTop: 4 }}>
                <button onClick={confirmDelete} disabled={deleting}
                  style={{ ...S.btn("danger"), flex: 1, justifyContent: "center", opacity: deleting ? 0.6 : 1 }}>
                  {deleting ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} /> : <Trash2 size={13} />} Delete
                </button>
                <button onClick={() => setDeleteConfirm(null)} style={{ ...S.btn("ghost"), flex: 1, justifyContent: "center" }}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════ QUICK EMAIL MODAL ══════════ */}
      {showEmailModal && (
        <div style={{ position: "fixed", inset: 0, zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
          onClick={e => { if (e.target === e.currentTarget) { setShowEmailModal(false); setEmailResult(null); } }}>
          <div style={{ ...S.card, width: 440, padding: 0, overflow: "hidden", boxShadow: "0 24px 80px rgba(0,0,0,0.4)" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>Quick Email</h3>
              <button onClick={() => { setShowEmailModal(false); setEmailResult(null); }} style={{ background: "none", border: "none", cursor: "pointer", color: "hsl(var(--muted-foreground))", padding: 4 }}><X size={16} /></button>
            </div>
            <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Subject</label>
                <input value={emailSubject} onChange={e => setEmailSubject(e.target.value)} placeholder="Subject…" style={S.input} />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>Message</label>
                <textarea value={emailBody} onChange={e => setEmailBody(e.target.value)} rows={5} placeholder="Write your message…" style={{ ...S.input, resize: "vertical", fontFamily: "inherit" }} />
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button onClick={async () => { await sendEmail(); }} disabled={emailSending || !emailSubject || !emailBody}
                  style={{ ...S.btn("primary"), opacity: emailSending || !emailSubject || !emailBody ? 0.5 : 1 }}>
                  {emailSending ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} /> : <Send size={13} />}
                  Send
                </button>
                {emailResult && <span style={{ fontSize: 12, color: "#22c55e" }}>{emailResult}</span>}
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>
    </div>
  );
}
