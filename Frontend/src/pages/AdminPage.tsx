import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Shield, Users, Activity, Trash2, Edit3, Check, X,
  RefreshCw, LogOut, Search, Terminal, Zap,
  TrendingUp, MessageSquare, Crown, User, AlertCircle,
  Eye, EyeOff, ArrowLeft, Wifi, Circle,
  Database, Cpu, Radio, Server, CheckCircle2, XCircle, AlertTriangle,
  BarChart3, Clock, ChevronUp, ChevronDown, Mail, Sun, Moon, Send,
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

interface LLMProviders {
  cf_primary?: string;
  cf_backup?:  string;
  groq?:       string;
  nvidia_nim?: string;
}

interface NewsPipelineInfo {
  available:       boolean;
  running?:        boolean;
  last_run_at?:    string | null;
  next_run_at?:    string | null;
  total_pages?:    number;
  total_items?:    number;
  total_articles?: number;
  status?:         string;
  error?:          string;
}

interface HealthData {
  status:       string;
  timestamp:    string;
  vector_store: string;
  services?:    string;
  neo4j?:       string;
  api_server?:  string;

  // LLM
  llm_status?:    string;
  llm_providers?: LLMProviders;

  // Voice
  voice_agent?:  string;
  elevenlabs?:   string;

  // Agents
  cad_ifc_agent?:  string;
  ingestion_mode?: string;
  wiki_enricher?:  string;
  llm_judge?:      string;

  // News
  news_pipeline?: NewsPipelineInfo;

  // Stats
  statistics?: {
    total_chunks?:     number;
    total_documents?:  number;
    active_sessions?:  number;
    cached_documents?: number;
  };
  active_sessions?: number;
  report_count?:    number;
}

interface LogEntry { ts: string; msg: string; }

// ── Structured pipeline log types ─────────────────────────────────────────────
type LogMode = "console" | "pipeline";
type PipelineEventType = "routing" | "judge" | "retrieval" | "ingestion" | "query_end" | "alert" | "latency";

interface PipelineLogEntry {
  ts:           string;
  event:        PipelineEventType;
  // routing
  session_id?:  string;
  query?:       string;
  route?:       string;
  confidence?:  number;
  latency_ms?:  number;
  intent?:      string;
  prev_route?:  string;
  forced?:      boolean;
  // judge
  attempt?:     number;
  passed?:      boolean;
  score?:       number;
  reason?:      string;
  // retrieval
  top_k?:       number;
  result_count?: number;
  min_score?:   number;
  max_score?:   number;
  avg_score?:   number;
  reranker_used?: boolean;
  graph_hits?:  number;
  source?:      string;
  // ingestion
  doc_id?:      string;
  filename?:    string;
  node?:        string;
  status?:      string;
  user_id?:     string;
  details?:     Record<string, unknown>;
  // query_end
  total_latency_ms?: number;
  success?:     boolean;
  judge_attempts?: number;
  sources_count?: number;
  error?:       string;
  // alert
  alert_type?:  string;
  message?:     string;
  // latency
  label?:       string;
  // obs stats
  event_counts?: Record<string, number>;
  judge_pass?:  number;
  judge_fail?:  number;
  judge_pass_rate?: number | null;
  log_file?:    string;
}

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
  return Date.now() - new Date(last_seen).getTime() < 5 * 60 * 1000;
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

// ─── Mini Sparkline Chart ─────────────────────────────────────────────────────

function Sparkline({ data, color, height = 36 }: { data: number[]; color: string; height?: number }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  if (!data.length) return null;
  const max = Math.max(...data, 1);
  const w = 120, h = height;
  const points = data.map((v, i) => ({
    x: (i / Math.max(data.length - 1, 1)) * w,
    y: h - (v / max) * h * 0.85 - 2,
    v,
  }));
  const pts = points.map(p => `${p.x},${p.y}`).join(" ");
  const areaPath = `M 0,${h} L ${points.map(p => `${p.x},${p.y}`).join(" L ")} L ${w},${h} Z`;
  const gradId = `sg-${color.replace("#", "")}`;
  return (
    <svg width={w} height={h} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#${gradId})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      {/* invisible hit-area strips for hover */}
      {points.map((p, i) => {
        const prev = points[i - 1];
        const next = points[i + 1];
        const x0 = prev ? (prev.x + p.x) / 2 : 0;
        const x1 = next ? (p.x + next.x) / 2 : w;
        return (
          <rect
            key={i}
            x={x0} y={0} width={x1 - x0} height={h}
            fill="transparent"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{ cursor: "crosshair" }}
          />
        );
      })}
      {/* dots — always show last, show hovered */}
      {points.map((p, i) => {
        const isLast = i === points.length - 1;
        const isHov  = i === hoveredIdx;
        if (!isLast && !isHov) return null;
        return (
          <circle
            key={i}
            cx={p.x} cy={p.y} r={isHov ? 4 : 3}
            fill={color}
            style={{ transition: "r 0.1s" }}
          />
        );
      })}
      {/* tooltip on hover */}
      {hoveredIdx !== null && (() => {
        const p = points[hoveredIdx];
        const tipW = 32, tipH = 18, pad = 6;
        let tx = p.x - tipW / 2;
        let ty = p.y - tipH - pad;
        if (tx < 0) tx = 0;
        if (tx + tipW > w) tx = w - tipW;
        if (ty < 0) ty = p.y + pad;
        return (
          <g style={{ pointerEvents: "none" }}>
            <rect x={tx} y={ty} width={tipW} height={tipH} rx={4}
              fill="hsl(220 15% 12%)" stroke={color} strokeWidth={0.8} opacity={0.95} />
            <text x={tx + tipW / 2} y={ty + tipH / 2 + 4.5}
              textAnchor="middle" fontSize={10} fontWeight={700} fill={color}>
              {p.v}
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

// ─── Bar Chart ────────────────────────────────────────────────────────────────

function MiniBarChart({ data, labels, color }: { data: number[]; labels: string[]; color: string }) {
  const max = Math.max(...data, 1);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 5, flex: 1, minHeight: 0 }}>
      {data.map((v, i) => {
        const pct = Math.max((v / max) * 100, v > 0 ? 8 : 2);
        return (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 5, height: "100%" }}>
            <div style={{ flex: 1, width: "100%", display: "flex", alignItems: "flex-end" }}>
              <div style={{
                width: "100%", borderRadius: "4px 4px 0 0",
                height: `${pct}%`,
                background: v > 0
                  ? `linear-gradient(to top, ${color}, ${color}88)`
                  : "hsl(var(--muted)/0.25)",
                transition: "height 0.5s cubic-bezier(.4,0,.2,1)",
                minHeight: v > 0 ? 4 : 2,
              }} title={`${labels[i]}: ${v}`} />
            </div>
            <span style={{ fontSize: 9, color: "hsl(var(--muted-foreground))", whiteSpace: "nowrap", flexShrink: 0 }}>{labels[i]}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Status Badge ──────────────────────────────────────────────────────────────

function ServiceBadge({ label, status, icon: Icon, detail }: {
  label: string; status: "ok" | "warn" | "error" | "unknown"; icon: React.ElementType; detail?: string;
}) {
  const colors = {
    ok:      { dot: "#22c55e", bg: "#22c55e15", border: "#22c55e30", text: "#22c55e" },
    warn:    { dot: "#f59e0b", bg: "#f59e0b15", border: "#f59e0b30", text: "#f59e0b" },
    error:   { dot: "#ef4444", bg: "#ef444415", border: "#ef444430", text: "#ef4444" },
    unknown: { dot: "#64748b", bg: "#64748b12", border: "#64748b25", text: "#64748b" },
  };
  const c = colors[status];
  const StatusIcon = status === "ok" ? CheckCircle2 : status === "error" ? XCircle : AlertTriangle;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
      background: c.bg, border: `1px solid ${c.border}`, borderRadius: 10,
    }}>
      <Icon size={15} color={c.text} />
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "hsl(var(--foreground))" }}>{label}</div>
        {detail && <div style={{ fontSize: 10, color: "hsl(var(--muted-foreground))", marginTop: 1 }}>{detail}</div>}
      </div>
      <StatusIcon size={14} color={c.text} />
    </div>
  );
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

function ExpandedChart({ data: rawData, color, label }: { data: number[]; color: string; label: string }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Sanitize: replace non-finite values with 0, require at least 1 point
  const data = (rawData ?? []).map(v => (typeof v === "number" && isFinite(v) ? v : 0));
  if (!data.length) return null;

  const W = 260, H = 90;
  const padL = 28, padR = 10, padT = 10, padB = 22;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  // Safely find peak index; fall back to 0 if not found
  const rawPeakIdx = data.indexOf(max);
  const peakIdx = rawPeakIdx >= 0 ? rawPeakIdx : 0;

  const toX = (i: number) => padL + (i / Math.max(data.length - 1, 1)) * chartW;
  const toY = (v: number) => padT + chartH - ((v - min) / range) * chartH;

  const points = data.map((v, i) => ({ x: toX(i), y: toY(v), v }));
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x},${p.y}`).join(" ");
  const areaPath = `${linePath} L ${points[points.length - 1].x},${padT + chartH} L ${points[0].x},${padT + chartH} Z`;

  // Y axis ticks
  const yTicks = [0, 0.5, 1].map(t => ({ v: Math.round(min + t * range), y: padT + chartH * (1 - t) }));
  // X axis labels — show first, mid, last (deduplicated for small datasets)
  const xLabels = [...new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])];
  const gradId = `exp-${color.replace(/[^a-z0-9]/gi, "")}`;

  return (
    <svg width={W} height={H} style={{ overflow: "visible", display: "block" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>

      {/* Grid lines */}
      {yTicks.map((t, i) => (
        <line key={i} x1={padL} y1={t.y} x2={W - padR} y2={t.y}
          stroke="rgba(255,255,255,0.06)" strokeWidth={1} strokeDasharray="3,3" />
      ))}

      {/* Y axis labels */}
      {yTicks.map((t, i) => (
        <text key={i} x={padL - 5} y={t.y + 4} textAnchor="end"
          fontSize={8} fill="rgba(148,163,184,0.7)" fontFamily="inherit">
          {t.v}
        </text>
      ))}

      {/* X axis labels */}
      {xLabels.map((idx) => (
        <text key={idx} x={toX(idx)} y={H - 4} textAnchor="middle"
          fontSize={8} fill="rgba(148,163,184,0.7)" fontFamily="inherit">
          {idx === 0 ? "7d ago" : idx === data.length - 1 ? "now" : "mid"}
        </text>
      ))}

      {/* Axis line */}
      <line x1={padL} y1={padT} x2={padL} y2={padT + chartH}
        stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
      <line x1={padL} y1={padT + chartH} x2={W - padR} y2={padT + chartH}
        stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

      {/* Area fill */}
      <path d={areaPath} fill={`url(#${gradId})`} />

      {/* Line */}
      <path d={linePath} fill="none" stroke={color} strokeWidth={2}
        strokeLinecap="round" strokeLinejoin="round" />

      {/* Peak marker */}
      <circle cx={points[peakIdx].x} cy={points[peakIdx].y} r={4}
        fill={color} stroke="hsl(220 15% 12%)" strokeWidth={2} />
      <text x={points[peakIdx].x} y={points[peakIdx].y - 8}
        textAnchor="middle" fontSize={8} fontWeight={700} fill={color} fontFamily="inherit">
        peak: {max}
      </text>

      {/* Hover hit areas */}
      {points.map((p, i) => {
        const prev = points[i - 1];
        const next = points[i + 1];
        const x0 = prev ? (prev.x + p.x) / 2 : padL;
        const x1 = next ? (p.x + next.x) / 2 : W - padR;
        return (
          <rect key={i} x={x0} y={padT} width={x1 - x0} height={chartH}
            fill="transparent"
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{ cursor: "crosshair" }} />
        );
      })}

      {/* Hovered dot + tooltip */}
      {hoveredIdx !== null && hoveredIdx !== peakIdx && (
        <circle cx={points[hoveredIdx].x} cy={points[hoveredIdx].y} r={3.5}
          fill={color} stroke="hsl(220 15% 12%)" strokeWidth={2} />
      )}
      {hoveredIdx !== null && (() => {
        const p = points[hoveredIdx];
        const tipW = 36, tipH = 18, gap = 6;
        let tx = p.x - tipW / 2;
        let ty = p.y - tipH - gap;
        if (tx < 0) tx = 0;
        if (tx + tipW > W) tx = W - tipW;
        if (ty < padT) ty = p.y + gap;
        return (
          <g style={{ pointerEvents: "none" }}>
            <rect x={tx} y={ty} width={tipW} height={tipH} rx={4}
              fill="hsl(220 15% 10%)" stroke={color} strokeWidth={0.8} opacity={0.96} />
            <text x={tx + tipW / 2} y={ty + tipH / 2 + 4.5}
              textAnchor="middle" fontSize={9} fontWeight={700} fill={color} fontFamily="inherit">
              {p.v}
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

function KpiCard({ icon: Icon, label, value, sub, accent, trend, sparkData }: {
  icon: React.ElementType; label: string; value: number | string; sub?: string; accent: string;
  trend?: number; sparkData?: number[];
}) {
  const [hovered, setHovered]   = useState(false);
  const [leaving, setLeaving]   = useState(false);
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleEnter = () => {
    if (leaveTimer.current) clearTimeout(leaveTimer.current);
    setLeaving(false);
    setHovered(true);
  };

  const handleLeave = () => {
    // Start the out-animation, then collapse after it finishes
    setLeaving(true);
    leaveTimer.current = setTimeout(() => {
      setHovered(false);
      setLeaving(false);
    }, 280); // matches kpiChartFadeOut duration
  };

  // True while the expanded state (card + chart) should be visible
  const expanded = hovered || leaving;

  return (
    <div
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      style={{
        background: "hsl(var(--card))",
        border: `1px solid ${expanded ? accent + "50" : "hsl(var(--border))"}`,
        borderRadius: 16,
        padding: expanded ? "18px 20px 16px" : "18px 20px",
        display: "flex", flexDirection: "column", gap: 6,
        position: "relative", overflow: "hidden",
        cursor: "default",
        minWidth: expanded ? 300 : undefined,
        boxShadow: expanded ? `0 8px 32px ${accent}22, 0 0 0 1px ${accent}20` : "none",
        transition: "min-width 0.35s cubic-bezier(0,0,.2,1), box-shadow 0.35s cubic-bezier(0,0,.2,1), border-color 0.3s ease, padding 0.25s ease",
        zIndex: expanded ? 10 : 1,
      }}
    >
      {/* Glow orb */}
      <div style={{
        position: "absolute", top: 0, right: 0,
        width: expanded ? 160 : 80, height: expanded ? 160 : 80,
        background: `radial-gradient(circle at 70% 30%, ${accent}${expanded ? "30" : "22"}, transparent 70%)`,
        pointerEvents: "none",
        transition: "width 0.35s cubic-bezier(0,0,.2,1), height 0.35s cubic-bezier(0,0,.2,1)",
      }} />

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8, background: `${accent}18`,
            border: `1px solid ${accent}30`, display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Icon size={14} color={accent} />
          </div>
          <span style={{ fontSize: 10, fontWeight: 600, color: "hsl(var(--muted-foreground))", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            {label}
          </span>
        </div>
        {trend !== undefined && (
          <div style={{ display: "flex", alignItems: "center", gap: 2, fontSize: 11, fontWeight: 600, color: trend >= 0 ? "#22c55e" : "#ef4444" }}>
            {trend >= 0 ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {Math.abs(trend)}%
          </div>
        )}
      </div>

      {/* Value */}
      <div style={{ fontSize: 26, fontWeight: 800, color: "hsl(var(--foreground))", lineHeight: 1.1 }}>{value}</div>

      {/* Chart area */}
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
        {sub && !expanded && <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }}>{sub}</div>}
        {sparkData && (
          <div style={{
            width: expanded ? 260 : 120,
            overflow: "hidden",
            transition: "width 0.35s cubic-bezier(0,0,.2,1)",
          }}>
            {expanded
              ? (
                <div style={{
                  animation: leaving
                    ? "kpiChartFadeOut 0.26s cubic-bezier(.4,0,1,1) forwards"
                    : "kpiChartFadeIn 0.25s cubic-bezier(0,0,.2,1) forwards",
                }}>
                  <div style={{ fontSize: 9, color: "hsl(var(--muted-foreground))", marginBottom: 4, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                    Last 7 days · {label}
                  </div>
                  <ExpandedChart data={sparkData} color={accent} label={label} />
                  {sub && <div style={{ fontSize: 10, color: "hsl(var(--muted-foreground))", marginTop: 6 }}>{sub}</div>}
                </div>
              )
              : <Sparkline data={sparkData} color={accent} />
            }
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

// ─── Shared styles (module-level so all components can access) ────────────────
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
    boxSizing: "border-box" as const,
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
  sectionTitle: {
    fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
    textTransform: "uppercase" as const, color: "hsl(var(--muted-foreground))",
    marginBottom: 10,
  },
};

// ─── News Pipeline Card ───────────────────────────────────────────────────────

function NewsPipelineCard({ health, onTrigger }: { health: HealthData | null; onTrigger: () => void }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const np = health?.news_pipeline;
  const isRunning = np?.running ?? false;

  const nextRun = np?.next_run_at ? new Date(np.next_run_at).getTime() : null;
  const lastRun = np?.last_run_at ? new Date(np.last_run_at).getTime() : null;

  const msLeft = nextRun ? Math.max(0, nextRun - now) : null;
  const countdown = (() => {
    if (msLeft === null) return "—";
    if (msLeft === 0) return "Running soon";
    const d = Math.floor(msLeft / 86400000);
    const h = Math.floor((msLeft % 86400000) / 3600000);
    const m = Math.floor((msLeft % 3600000) / 60000);
    const s = Math.floor((msLeft % 60000) / 1000);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
  })();

  const totalArticles = np?.total_articles ?? (np?.total_items) ?? 0;
  const totalPages = np?.total_pages ?? 0;
  const statusColor = isRunning ? "#f59e0b" : np?.available ? "#22c55e" : "#6b7280";
  const statusLabel = isRunning ? "Running" : np?.available ? (np?.status === "ok" ? "Ready" : np?.status ?? "Ready") : "Unavailable";

  return (
    <div style={{ ...S.card, padding: "18px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 7, background: "#f59e0b15", border: "1px solid #f59e0b25", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Clock size={13} color="#f59e0b" />
          </div>
          <span style={{ fontSize: 12, fontWeight: 700, color: "hsl(var(--foreground))", textTransform: "uppercase", letterSpacing: "0.06em" }}>News Pipeline</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: statusColor, animation: isRunning ? "pulse 1s infinite" : "none" }} />
          <span style={{ fontSize: 10, fontWeight: 600, color: statusColor, textTransform: "uppercase" }}>{statusLabel}</span>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
        {[
          { label: "Articles Cached", value: totalArticles.toLocaleString(), color: "#3b82f6" },
          { label: "Pages",           value: totalPages.toString(),           color: "#a855f7" },
        ].map(s => (
          <div key={s.label} style={{ background: "hsl(var(--muted)/0.3)", borderRadius: 8, padding: "10px 12px", border: "1px solid hsl(var(--border))" }}>
            <div style={{ fontSize: 18, fontWeight: 800, color: s.color, lineHeight: 1 }}>{s.value || "—"}</div>
            <div style={{ fontSize: 10, color: "hsl(var(--muted-foreground))", marginTop: 3 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Timeline */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
          <span style={{ color: "hsl(var(--muted-foreground))" }}>Last run</span>
          <span style={{ color: "hsl(var(--foreground))", fontWeight: 600 }}>{lastRun ? timeAgo(np?.last_run_at!) : "Never"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
          <span style={{ color: "hsl(var(--muted-foreground))" }}>Next run in</span>
          <span style={{ color: msLeft !== null && msLeft < 3600000 ? "#f59e0b" : "hsl(var(--foreground))", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{countdown}</span>
        </div>
        {/* Progress bar */}
        {lastRun && nextRun && (
          <div style={{ height: 3, borderRadius: 999, background: "hsl(var(--muted)/0.4)", overflow: "hidden", marginTop: 2 }}>
            <div style={{
              height: "100%", borderRadius: 999,
              background: "linear-gradient(to right, #f59e0b, #ef4444)",
              width: `${Math.min(100, ((now - lastRun) / (nextRun - lastRun)) * 100)}%`,
              transition: "width 1s linear",
            }} />
          </div>
        )}
      </div>

      {/* Trigger button */}
      <button
        onClick={onTrigger}
        style={{ ...S.btn("ghost"), width: "100%", justifyContent: "center", gap: 6, fontSize: 11, padding: "7px 12px", border: "1px solid hsl(var(--border))" }}
      >
        <RefreshCw size={11} />
        Trigger Manual Refresh
      </button>
    </div>
  );
}

export default function AdminPage() {
  const { currentUser, logout, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab]           = useState<"users" | "health" | "logs" | "settings">("users");
  const [users, setUsers]       = useState<AdminUser[]>([]);
  const [stats, setStats]       = useState<Stats | null>(null);
  const [health, setHealth]     = useState<HealthData | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [logs, setLogs]         = useState<LogEntry[]>([]);
  const [search, setSearch]     = useState("");
  const [loading, setLoading]   = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);

  // Edit modal
  const [editUser, setEditUser]         = useState<AdminUser | null>(null);
  const [editUsername, setEditUsername] = useState("");
  const [editEmail, setEditEmail]       = useState("");
  const [editPassword, setEditPassword] = useState("");
  const [editRole, setEditRole]         = useState<"user" | "admin">("user");
  const [showPw, setShowPw]             = useState(false);
  const [editSaving, setEditSaving]     = useState(false);

  // Delete confirm
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleting, setDeleting]           = useState(false);

  // Self-credentials panel
  const [selfUsername, setSelfUsername] = useState(currentUser?.username || "");
  const [selfEmail, setSelfEmail]       = useState(currentUser?.email || "");
  const [selfPassword, setSelfPassword] = useState("");
  const [selfSaving, setSelfSaving]     = useState(false);
  const [selfMsg, setSelfMsg]           = useState<string | null>(null);

  const logEndRef = useRef<HTMLDivElement>(null);
  const esRef     = useRef<EventSource | null>(null);

  // ── Log mode switch: console vs structured pipeline ────────────────────────
  const [logMode, setLogMode]               = useState<LogMode>("console");
  const [pipelineLogs, setPipelineLogs]     = useState<PipelineLogEntry[]>([]);
  const [pipelineFilter, setPipelineFilter] = useState<PipelineEventType | "all">("all");
  const [pipelineStats, setPipelineStats]   = useState<PipelineLogEntry | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const pipelineEndRef = useRef<HTMLDivElement>(null);

  // ── Dark mode ──────────────────────────────────────────────────────────────
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggleDark = () => {
    const isDark = document.documentElement.classList.toggle("dark");
    setDark(isDark);
    localStorage.setItem("theme", isDark ? "dark" : "light");
  };

  // ── Email compose modal ────────────────────────────────────────────────────
  const [emailModal, setEmailModal]   = useState(false);
  const [emailTarget, setEmailTarget] = useState<AdminUser | null>(null); // null = broadcast
  const [emailSubject, setEmailSubject] = useState("");
  const [emailBody, setEmailBody]     = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [emailResult, setEmailResult]   = useState<string | null>(null);

  async function sendEmail() {
    if (!currentUser?.token || !emailSubject.trim() || !emailBody.trim()) return;
    setEmailSending(true); setEmailResult(null);
    const ids = emailTarget ? [emailTarget.user_id] : users.map(u => u.user_id);
    try {
      const res = await fetch(`${API}/auth/admin/send-email`, {
        method: "POST",
        headers: authHeaders(currentUser.token),
        body: JSON.stringify({ user_ids: ids, subject: emailSubject, body: emailBody }),
      });
      const data = await res.json();
      if (res.ok) {
        setEmailResult(`✅ Sent to ${data.sent?.length ?? 0} user(s)${data.failed?.length ? `, ${data.failed.length} failed` : ""}`);
        setTimeout(() => { setEmailModal(false); setEmailSubject(""); setEmailBody(""); setEmailResult(null); }, 2000);
      } else {
        setEmailResult(`❌ ${data.detail || "Failed to send"}`);
      }
    } catch { setEmailResult("❌ Network error"); }
    finally { setEmailSending(false); }
  }

  // ── Auth guard ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!currentUser) { navigate("/"); return; }
    if (currentUser.role !== "admin") { navigate("/"); return; }
  }, [currentUser, navigate]);

  // ── Ensure avatar_url is populated (Google OAuth stores it in Neo4j) ────────
  const [adminAvatar, setAdminAvatar] = useState<string | null>(currentUser?.avatar_url || null);
  useEffect(() => {
    if (!currentUser?.token) return;
    // Re-fetch profile to get avatar_url that may not be in the cached token
    fetch(`${API}/auth/me`, { headers: { Authorization: `Bearer ${currentUser.token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.avatar_url) setAdminAvatar(d.avatar_url); })
      .catch(() => {});
  }, [currentUser?.token]);

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

  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const res = await fetch(`${API}/health`);
      if (res.ok) setHealth(await res.json());
    } finally { setHealthLoading(false); }
  }, []);

  useEffect(() => {
    loadUsers();
    loadStats();
    loadHealth();
  }, [loadUsers, loadStats, loadHealth]);

  // Auto-refresh health every 30s
  useEffect(() => {
    const interval = setInterval(loadHealth, 30_000);
    return () => clearInterval(interval);
  }, [loadHealth]);

  // ── Live log SSE ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (tab !== "logs" || !currentUser?.token) return;
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
    fetch(`${API}/auth/admin/logs?limit=200`, { headers: authHeaders(currentUser.token) })
      .then(r => r.json()).then(d => setLogs(d.logs || [])).catch(() => {});
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

  // ── Pipeline log fetch (structured JSON) ────────────────────────────────────
  const loadPipelineLogs = useCallback(async () => {
    if (!currentUser?.token) return;
    setPipelineLoading(true);
    try {
      const [logsRes, statsRes] = await Promise.all([
        fetch(`${API}/auth/admin/pipeline-logs?limit=300`, { headers: authHeaders(currentUser.token) }),
        fetch(`${API}/auth/admin/pipeline-logs/stats`,     { headers: authHeaders(currentUser.token) }),
      ]);
      if (logsRes.ok) {
        const d = await logsRes.json();
        setPipelineLogs(d.entries || []);
      }
      if (statsRes.ok) setPipelineStats(await statsRes.json());
    } catch {}
    finally { setPipelineLoading(false); }
  }, [currentUser?.token]);

  useEffect(() => {
    if (tab === "logs" && logMode === "pipeline") loadPipelineLogs();
  }, [tab, logMode, loadPipelineLogs]);


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

  // ── Derived data for charts ─────────────────────────────────────────────────
  const filtered     = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  );
  const onlineCount  = users.filter(u => isOnline(u.last_seen)).length;
  const offlineCount = users.length - onlineCount;

  // Build sparkline data from users (conversations per-user distribution)
  const convSparkData = users.slice(0, 12).map(u => u.conversation_count);
  const docSparkData  = users.slice(0, 12).map(u => u.document_count);

  // Activity bar data: bucket users by signup recency (last 7 days)
  const activityBars = (() => {
    const days = ["6d","5d","4d","3d","2d","1d","Today"];
    const counts = new Array(7).fill(0);
    users.forEach(u => {
      if (!u.created_at) return;
      const diff = Math.floor((Date.now() - new Date(u.created_at).getTime()) / 86_400_000);
      if (diff < 7) counts[6 - diff]++;
    });
    return { days, counts };
  })();

  // Conversation leader board: top 5 users
  const topUsers = [...users].sort((a, b) => b.conversation_count - a.conversation_count).slice(0, 5);

  // Health service status helper
  function deriveStatus(val: string | undefined): "ok" | "warn" | "error" | "unknown" {
    if (!val) return "unknown";
    const v = val.toLowerCase();
    if (["ok", "healthy", "connected", "ready", "configured", "available"].includes(v)) return "ok";
    if (["warn", "degraded", "not_configured", "unavailable", "fallback_direct"].includes(v)) return "warn";
    if (["error", "down", "failed", "disconnected", "not_available"].includes(v)) return "error";
    return "unknown";
  }

  // While auth context is still hydrating, show a neutral loading screen
  // instead of returning null (which would cause a blank flash on first navigation).
  if (authLoading) {
    return (
      <div style={{ ...S.page, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
          <RefreshCw size={24} style={{ animation: "spin 0.8s linear infinite", color: "hsl(var(--muted-foreground))" }} />
          <span style={{ fontSize: 13, color: "hsl(var(--muted-foreground))" }}>Loading…</span>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!currentUser || currentUser.role !== "admin") return null;

  // ── Styles ──────────────────────────────────────────────────────────────────
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
            {/* Platform logo */}
            <img src="/favicon.svg" alt="Bimlo" style={{ width: 26, height: 26, borderRadius: 6 }} />
            <span style={{ fontSize: 15, fontWeight: 800, color: "hsl(var(--foreground))" }}>Admin Dashboard</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* Online/offline pill */}
          <div style={{
            display: "flex", alignItems: "center", gap: 8, fontSize: 12,
            padding: "4px 10px", borderRadius: 999,
            background: "hsl(var(--muted)/0.4)",
            border: "1px solid hsl(var(--border))",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Circle size={6} fill="#22c55e" color="#22c55e" />
              <span style={{ color: "#22c55e", fontWeight: 700 }}>{onlineCount}</span>
            </div>
            <span style={{ color: "hsl(var(--muted-foreground))" }}>online</span>
          </div>

          {/* Current admin avatar */}
          {adminAvatar ? (
            <img
              src={adminAvatar}
              referrerPolicy="no-referrer"
              crossOrigin="anonymous"
              alt={currentUser.username}
              style={{ width: 30, height: 30, borderRadius: "50%", objectFit: "cover", border: "2px solid #7c3aed55" }}
              onError={() => setAdminAvatar(null)}
            />
          ) : (
            <div style={{
              width: 30, height: 30, borderRadius: "50%",
              background: "linear-gradient(135deg,#7c3aed,#4f46e5)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 12, fontWeight: 700, color: "#fff", border: "2px solid #7c3aed55",
            }}>
              {currentUser.username[0]?.toUpperCase()}
            </div>
          )}

          {/* Dark mode toggle */}
          <button onClick={toggleDark} title="Toggle theme" style={{ ...S.btn("ghost"), padding: "6px 8px" }}>
            {dark ? <Sun size={14} /> : <Moon size={14} />}
          </button>

          <button onClick={() => { loadUsers(); loadStats(); loadHealth(); }} style={S.btn("ghost")}>
            <RefreshCw size={13} />
          </button>
          <button onClick={logout} style={S.btn("ghost")}>
            <LogOut size={13} /> Logout
          </button>
        </div>
      </div>

      <div style={{ padding: "24px 28px", maxWidth: 1360, margin: "0 auto" }}>

        {/* ── KPIs ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 14, marginBottom: 24 }}>
          <KpiCard icon={Users}       label="Total Users"  value={stats?.total_users ?? "—"}    sub={`${stats?.admin_users ?? 0} admin(s)`}  accent="#3b82f6" sparkData={[2,3,3,4,5,5,6,7,7,stats?.total_users ?? 7]} />
          <KpiCard icon={Wifi}        label="Active (1h)"  value={stats?.active_1h ?? "—"}      sub="online now"        accent="#22c55e" sparkData={[1,0,2,1,3,2,stats?.active_1h ?? 2]} />
          <KpiCard icon={Activity}    label="Active (24h)" value={stats?.active_24h ?? "—"}     sub="last 24 hours"     accent="#06b6d4" sparkData={[1,2,1,3,2,4,stats?.active_24h ?? 2]} />
          <KpiCard icon={TrendingUp}  label="New (7d)"     value={stats?.new_users_7d ?? "—"}   sub="new signups"       accent="#a855f7" trend={stats?.new_users_7d ? 12 : undefined} sparkData={activityBars.counts} />
          <KpiCard icon={MessageSquare} label="Conversations" value={stats?.total_conversations ?? "—"} sub="all sessions" accent="#f59e0b" sparkData={convSparkData} />
        </div>

        {/* ── Activity + Top Users row ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
          {/* Signups per day */}
          <div style={{ ...S.card, padding: "18px 20px", display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexShrink: 0 }}>
              <div>
                <div style={{ ...S.sectionTitle, marginBottom: 2 }}>Signups — last 7 days</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: "hsl(var(--foreground))" }}>
                  {activityBars.counts.reduce((a, b) => a + b, 0)}
                  <span style={{ fontSize: 12, fontWeight: 500, color: "hsl(var(--muted-foreground))", marginLeft: 6 }}>new users</span>
                </div>
              </div>
              <BarChart3 size={18} color="#a855f7" />
            </div>
            <MiniBarChart data={activityBars.counts} labels={activityBars.days} color="#a855f7" />
          </div>

          {/* News Pipeline Status */}
          <NewsPipelineCard health={health} onTrigger={async () => {
            try {
              const { token } = JSON.parse(localStorage.getItem("bimlo_auth") || "{}");
              await fetch(`${API}/api/news/trigger`, { method: "POST", headers: authHeaders(token || "") });
              setTimeout(loadHealth, 2000);
            } catch {}
          }} />
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20 }}>
          {(["users", "health", "logs", "settings"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} style={S.tab(tab === t)}>
              {t === "users"   && <><Users size={13} /> Users</>}
              {t === "health"  && <><Server size={13} /> System Health</>}
              {t === "logs"    && <><Terminal size={13} /> Live Logs</>}
              {t === "settings"&& <><Shield size={13} /> My Account</>}
            </button>
          ))}
        </div>

        {/* ══════════════════════ USERS TAB ══════════════════════ */}
        {tab === "users" && (
          <div style={S.card}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ position: "relative", flex: 1, maxWidth: 320 }}>
                <Search size={13} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "hsl(var(--muted-foreground))" }} />
                <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search users…" style={{ ...S.input, paddingLeft: 30 }} />
              </div>
              <span style={{ fontSize: 12, color: "hsl(var(--muted-foreground))", marginLeft: "auto" }}>
                {filtered.length} user{filtered.length !== 1 ? "s" : ""}
              </span>
              <button onClick={() => { setEmailTarget(null); setEmailSubject(""); setEmailBody(""); setEmailResult(null); setEmailModal(true); }}
                style={{ ...S.btn("ghost"), gap: 6, border: "1px solid hsl(var(--border))" }}>
                <Mail size={12} /> Broadcast
              </button>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid hsl(var(--border))" }}>
                    {["User", "Email", "Role", "Status", "Conversations", "Documents", "Joined", "Last seen", "Actions"].map(h => (
                      <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", letterSpacing: "0.05em", textTransform: "uppercase", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    <tr><td colSpan={9} style={{ padding: 40, textAlign: "center", color: "hsl(var(--muted-foreground))" }}>
                      <RefreshCw size={18} style={{ animation: "spin 0.8s linear infinite", margin: "0 auto" }} />
                    </td></tr>
                  ) : filtered.length === 0 ? (
                    <tr><td colSpan={9} style={{ padding: 40, textAlign: "center", color: "hsl(var(--muted-foreground))", fontSize: 13 }}>No users found</td></tr>
                  ) : filtered.map(u => {
                    const online = isOnline(u.last_seen);
                    const isMe   = u.user_id === currentUser?.user_id;
                    return (
                      <tr key={u.user_id}
                        style={{ borderBottom: "1px solid hsl(var(--border)/0.5)", background: isMe ? "hsl(var(--primary)/0.04)" : "transparent", transition: "background 0.1s" }}
                        onMouseEnter={e => (e.currentTarget as HTMLTableRowElement).style.background = isMe ? "hsl(var(--primary)/0.07)" : "hsl(var(--muted)/0.3)"}
                        onMouseLeave={e => (e.currentTarget as HTMLTableRowElement).style.background = isMe ? "hsl(var(--primary)/0.04)" : "transparent"}
                      >
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            {/* Avatar with Google profile pic support */}
                            <div style={{
                              width: 34, height: 34, borderRadius: "50%", flexShrink: 0,
                              background: u.avatar_url ? "transparent" : (u.role === "admin" ? "linear-gradient(135deg,#7c3aed,#4f46e5)" : "linear-gradient(135deg,#3b82f6,#06b6d4)"),
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: 13, fontWeight: 700, color: "#fff", overflow: "hidden",
                              border: u.role === "admin" ? "2px solid #7c3aed44" : "2px solid #3b82f644",
                              boxShadow: online ? `0 0 0 2px ${u.role === "admin" ? "#7c3aed" : "#22c55e"}44` : "none",
                            }}>
                              {u.avatar_url
                                ? <img src={u.avatar_url} alt="" referrerPolicy="no-referrer" style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                    onError={e => { (e.target as HTMLImageElement).style.display = "none"; }} />
                                : u.username[0]?.toUpperCase()
                              }
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
                            {u.role === "admin" ? <Crown size={9} /> : <User size={9} />} {u.role}
                          </span>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12 }}>
                            <Circle size={7} fill={online ? "#22c55e" : "#64748b"} color={online ? "#22c55e" : "#64748b"} />
                            <span style={{ color: online ? "#22c55e" : "hsl(var(--muted-foreground))" }}>{online ? "Online" : "Offline"}</span>
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 13, color: "hsl(var(--foreground))", textAlign: "center" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "center" }}>
                            <MessageSquare size={11} color="#f59e0b" />
                            {u.conversation_count}
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 13, color: "hsl(var(--foreground))", textAlign: "center" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "center" }}>
                            <Zap size={11} color="#06b6d4" />
                            {u.document_count}
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 12, color: "hsl(var(--muted-foreground))", whiteSpace: "nowrap" }}>
                          {u.created_at ? timeAgo(u.created_at) : "—"}
                        </td>
                        <td style={{ padding: "12px 16px", fontSize: 12, color: "hsl(var(--muted-foreground))", whiteSpace: "nowrap" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                            <Clock size={10} /> {timeAgo(u.last_seen)}
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <button onClick={() => openEdit(u)} title="Edit" style={{
                              width: 28, height: 28, borderRadius: 6, border: "1px solid hsl(var(--border))",
                              background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                              color: "hsl(var(--muted-foreground))",
                            }}><Edit3 size={12} /></button>
                            <button onClick={() => { setEmailTarget(u); setEmailSubject(""); setEmailBody(""); setEmailResult(null); setEmailModal(true); }} title="Send message" style={{
                              width: 28, height: 28, borderRadius: 6, border: "1px solid #3b82f630",
                              background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                              color: "#3b82f6",
                            }}><Mail size={12} /></button>
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

        {/* ══════════════════════ HEALTH TAB ══════════════════════ */}
        {tab === "health" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 800, color: "hsl(var(--foreground))" }}>System Health</div>
                {health?.timestamp && (
                  <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))", marginTop: 2 }}>
                    Last checked: {new Date(health.timestamp).toLocaleTimeString()}
                  </div>
                )}
              </div>
              <button onClick={loadHealth} disabled={healthLoading} style={{ ...S.btn("ghost"), gap: 6 }}>
                <RefreshCw size={13} style={healthLoading ? { animation: "spin 0.8s linear infinite" } : {}} />
                Refresh
              </button>
            </div>

            {/* Overall status banner */}
            <div style={{
              padding: "14px 18px", borderRadius: 12,
              display: "flex", alignItems: "center", gap: 12,
              background: health?.status === "healthy" ? "linear-gradient(135deg, #22c55e10, #16a34a08)" : "#ef444410",
              border: `1px solid ${health?.status === "healthy" ? "#22c55e30" : "#ef444430"}`,
            }}>
              <div style={{ width: 36, height: 36, borderRadius: "50%", background: health?.status === "healthy" ? "#22c55e18" : "#ef444418", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {health?.status === "healthy" ? <CheckCircle2 size={18} color="#22c55e" /> : <XCircle size={18} color="#ef4444" />}
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>
                  {health?.status === "healthy" ? "All Systems Operational" : health ? "System Issues Detected" : "Checking system status…"}
                </div>
                <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))", marginTop: 1 }}>
                  {health?.services ?? "Loading service status…"}
                </div>
              </div>
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite" }} />
                <span style={{ fontSize: 11, fontWeight: 600, color: "#22c55e" }}>LIVE</span>
              </div>
            </div>

            {/* ── Row 1: Core Infrastructure + LLM Providers ── */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

              {/* Core Infrastructure */}
              <div style={{ ...S.card, padding: "18px 20px" }}>
                <div style={{ ...S.sectionTitle, marginBottom: 12 }}>Core Infrastructure</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <ServiceBadge label="API Server (FastAPI)"    status={health ? "ok" : "unknown"}              icon={Server}   detail="Backend running" />
                  <ServiceBadge label="Vector Store (Chroma)"   status={deriveStatus(health?.vector_store)}     icon={Cpu}      detail={health?.statistics?.total_chunks != null ? `${health.statistics.total_chunks.toLocaleString()} chunks · ${health.statistics.total_documents ?? 0} docs` : "Embedding & similarity search"} />
                  <ServiceBadge label="Neo4j Graph Database"    status={deriveStatus(health?.neo4j)}            icon={Database} detail={health?.neo4j === "connected" ? "Auth, sessions & knowledge graph" : health?.neo4j ?? "Not configured"} />
                  <ServiceBadge label="Voice Agent (ElevenLabs)" status={deriveStatus(health?.elevenlabs)}     icon={Radio}    detail={health?.elevenlabs === "configured" ? "TTS ready — eleven_flash_v2_5" : "ELEVENLABS_API_KEY not set"} />
                </div>
              </div>

              {/* LLM Providers */}
              <div style={{ ...S.card, padding: "18px 20px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                  <div style={{ ...S.sectionTitle }}>LLM Providers</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: health?.llm_status === "ok" ? "#22c55e" : "#ef4444" }} />
                    <span style={{ fontSize: 10, color: health?.llm_status === "ok" ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                      {health?.llm_status === "ok" ? "Operational" : "Degraded"}
                    </span>
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {([
                    { key: "cf_primary",  label: "CF Worker Primary",      icon: Zap,      color: "#f97316" },
                    { key: "cf_backup",   label: "CF Worker Backup",       icon: Zap,      color: "#f97316" },
                    { key: "groq",        label: "Groq (Llama 3.3 70B)",   icon: Cpu,      color: "#8b5cf6" },
                    { key: "nvidia_nim",  label: "NVIDIA NIM (MiniMax)",   icon: Activity, color: "#76b900" },
                  ] as const).map(p => {
                    const val = health?.llm_providers?.[p.key as keyof LLMProviders];
                    const st = val === "configured" ? "ok" : val === "not_configured" ? "warn" : "unknown";
                    return (
                      <ServiceBadge key={p.key} label={p.label} status={st} icon={p.icon}
                        detail={val === "configured" ? "API key configured" : "API key not set"} />
                    );
                  })}
                </div>
              </div>
            </div>

            {/* ── Row 2: Agents & Pipelines + Platform Metrics ── */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

              {/* Agents & Pipelines */}
              <div style={{ ...S.card, padding: "18px 20px" }}>
                <div style={{ ...S.sectionTitle, marginBottom: 12 }}>Agents & Pipelines</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <ServiceBadge
                    label="Ingestion Pipeline"
                    status={health?.ingestion_mode === "langgraph" ? "ok" : health?.ingestion_mode ? "warn" : "unknown"}
                    icon={Zap}
                    detail={health?.ingestion_mode === "langgraph" ? "LangGraph 3-node pipeline" : "Fallback: direct indexing"}
                  />
                  <ServiceBadge
                    label="LLM Judge (RAG Evaluator)"
                    status={deriveStatus(health?.llm_judge)}
                    icon={CheckCircle2}
                    detail={health?.llm_judge === "available" ? "Plan → generate → evaluate loop" : "Module not found"}
                  />
                  <ServiceBadge
                    label="Wiki Enricher"
                    status={deriveStatus(health?.wiki_enricher)}
                    icon={Search}
                    detail={health?.wiki_enricher === "available" ? "wikipedia-api installed" : "pip install wikipedia-api"}
                  />
                  <ServiceBadge
                    label="CAD / IFC Agent"
                    status={deriveStatus(health?.cad_ifc_agent)}
                    icon={BarChart3}
                    detail={health?.cad_ifc_agent === "available" ? "IFC, DXF, DWG, STEP supported" : "Module not loaded"}
                  />
                  <ServiceBadge
                    label="News Pipeline"
                    status={health?.news_pipeline?.available ? (health.news_pipeline.running ? "warn" : "ok") : "error"}
                    icon={Clock}
                    detail={health?.news_pipeline?.available
                      ? (health.news_pipeline.running ? "Currently running…" : `${health.news_pipeline.total_articles ?? health.news_pipeline.total_items ?? 0} articles cached`)
                      : "Module not found"}
                  />
                </div>
              </div>

              {/* Platform Metrics */}
              <div style={{ ...S.card, padding: "18px 20px" }}>
                <div style={{ ...S.sectionTitle, marginBottom: 12 }}>Platform Metrics</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {([
                    { label: "Total Users",         value: stats?.total_users ?? 0,               icon: Users,         color: "#3b82f6" },
                    { label: "Conversations",        value: stats?.total_conversations ?? 0,       icon: MessageSquare, color: "#f59e0b" },
                    { label: "Documents Uploaded",   value: stats?.total_documents ?? 0,           icon: Zap,           color: "#06b6d4" },
                    { label: "Reports Generated",    value: health?.report_count ?? stats?.total_reports ?? 0, icon: BarChart3, color: "#22c55e" },
                    { label: "Vector Chunks",        value: health?.statistics?.total_chunks ?? 0, icon: Database,      color: "#8b5cf6" },
                    { label: "Active Sessions",      value: health?.active_sessions ?? 0,          icon: Activity,      color: "#f97316" },
                  ] as const).map(m => (
                    <div key={m.label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 28, height: 28, borderRadius: 7, background: `${m.color}15`, border: `1px solid ${m.color}25`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <m.icon size={13} color={m.color} />
                      </div>
                      <span style={{ flex: 1, fontSize: 12, color: "hsl(var(--muted-foreground))" }}>{m.label}</span>
                      <span style={{ fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>{Number(m.value).toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════ LOGS TAB ══════════════════════ */}
        {tab === "logs" && (() => {
          // ── Pipeline event helpers ──────────────────────────────────────────
          const EVENT_META: Record<string, { label: string; color: string; icon: string }> = {
            routing:   { label: "Route",     color: "#3b82f6", icon: "🗺️" },
            judge:     { label: "Judge",     color: "#a855f7", icon: "⚖️" },
            retrieval: { label: "Retrieval", color: "#06b6d4", icon: "🔍" },
            ingestion: { label: "Ingest",    color: "#f59e0b", icon: "📦" },
            query_end: { label: "Query",     color: "#22c55e", icon: "✅" },
            alert:     { label: "Alert",     color: "#ef4444", icon: "🚨" },
            latency:   { label: "Latency",   color: "#64748b", icon: "⏱️" },
          };

          const filteredPipeline = pipelineLogs.filter(e =>
            pipelineFilter === "all" || e.event === pipelineFilter
          );

          function fmtPipelineRow(e: PipelineLogEntry) {
            const ts = e.ts ? new Date(e.ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
            const meta = EVENT_META[e.event] || { label: e.event, color: "#94a3b8", icon: "📝" };
            let summary = "";
            switch (e.event) {
              case "routing":
                summary = `→ ${e.route ?? "?"}  conf=${((e.confidence ?? 0) * 100).toFixed(0)}%  ${e.latency_ms != null ? e.latency_ms.toFixed(0) + "ms" : ""}${e.forced ? "  [FORCED]" : ""}`;
                break;
              case "judge":
                summary = `attempt=${e.attempt}  ${e.passed ? "✅ PASS" : "❌ FAIL"}  score=${((e.score ?? 0) * 100).toFixed(0)}%${e.reason ? "  " + e.reason.slice(0, 60) : ""}`;
                break;
              case "retrieval":
                summary = `hits=${e.result_count ?? 0}  avg=${((e.avg_score ?? 0)).toFixed(3)}  rerank=${e.reranker_used ? "yes" : "no"}  ${e.latency_ms != null ? e.latency_ms.toFixed(0) + "ms" : ""}`;
                break;
              case "ingestion":
                summary = `[${e.node ?? "?"}]  ${e.status ?? "?"}  ${e.filename ?? ""}  ${e.latency_ms != null ? e.latency_ms.toFixed(0) + "ms" : ""}`;
                break;
              case "query_end":
                summary = `${e.success ? "✅" : "❌"}  route=${e.route ?? "?"}  ${e.total_latency_ms != null ? e.total_latency_ms.toFixed(0) + "ms" : ""}  srcs=${e.sources_count ?? 0}  judge×${e.judge_attempts ?? 1}`;
                break;
              case "alert":
                summary = `[${e.alert_type ?? "?"}]  ${(e.message ?? "").slice(0, 100)}`;
                break;
              case "latency":
                summary = `${e.label ?? "?"}  ${e.latency_ms != null ? e.latency_ms.toFixed(1) + "ms" : ""}`;
                break;
              default:
                summary = JSON.stringify(e).slice(0, 120);
            }
            const sessionShort = e.session_id ? e.session_id.slice(-8) : "";
            return { ts, meta, summary, sessionShort };
          }

          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              {/* ── Mode switcher header ─────────────────────────────────── */}
              <div style={{ ...S.card, overflow: "hidden" }}>
                <div style={{
                  padding: "12px 20px", borderBottom: "1px solid hsl(var(--border))",
                  display: "flex", alignItems: "center", gap: 10,
                  background: "hsl(var(--muted)/0.2)",
                }}>
                  {/* Mode pills */}
                  <div style={{ display: "flex", gap: 3, background: "hsl(var(--muted)/0.4)", borderRadius: 9, padding: 3 }}>
                    {(["console", "pipeline"] as LogMode[]).map(m => (
                      <button key={m} onClick={() => setLogMode(m)} style={{
                        padding: "5px 13px", borderRadius: 7, fontSize: 12, fontWeight: 600,
                        border: "none", cursor: "pointer", transition: "all 0.15s",
                        background: logMode === m
                          ? (m === "console" ? "#22c55e" : "#3b82f6")
                          : "transparent",
                        color: logMode === m ? "#fff" : "hsl(var(--muted-foreground))",
                        display: "flex", alignItems: "center", gap: 5,
                      }}>
                        {m === "console" ? <Terminal size={11} /> : <Activity size={11} />}
                        {m === "console" ? "Console" : "Pipeline"}
                      </button>
                    ))}
                  </div>

                  {/* Console: live indicator + count + clear */}
                  {logMode === "console" && (<>
                    <div style={{ display: "flex", alignItems: "center", gap: 5, marginLeft: 8 }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite" }} />
                      <span style={{ fontSize: 11, color: "#22c55e", fontWeight: 600 }}>LIVE</span>
                    </div>
                    <span style={{ fontSize: 11, color: "hsl(var(--muted-foreground))", marginLeft: "auto" }}>{logs.length} entries</span>
                    <button onClick={() => setLogs([])} style={S.btn("ghost")}><X size={12} /> Clear</button>
                  </>)}

                  {/* Pipeline: filter chips + refresh */}
                  {logMode === "pipeline" && (<>
                    <div style={{ display: "flex", gap: 4, marginLeft: 8, flexWrap: "wrap" }}>
                      {(["all", "routing", "judge", "retrieval", "ingestion", "query_end", "alert"] as const).map(f => {
                        const m = f === "all" ? null : EVENT_META[f];
                        return (
                          <button key={f} onClick={() => setPipelineFilter(f)} style={{
                            padding: "3px 9px", borderRadius: 99, fontSize: 11, fontWeight: 600,
                            border: `1px solid ${pipelineFilter === f ? (m?.color ?? "#3b82f6") : "hsl(var(--border))"}`,
                            background: pipelineFilter === f ? `${m?.color ?? "#3b82f6"}18` : "transparent",
                            color: pipelineFilter === f ? (m?.color ?? "#3b82f6") : "hsl(var(--muted-foreground))",
                            cursor: "pointer",
                          }}>
                            {f === "all" ? "All" : `${m?.icon} ${m?.label}`}
                          </button>
                        );
                      })}
                    </div>
                    <button onClick={loadPipelineLogs} disabled={pipelineLoading} style={{ ...S.btn("ghost"), marginLeft: "auto", gap: 5 }}>
                      <RefreshCw size={11} style={pipelineLoading ? { animation: "spin 0.8s linear infinite" } : {}} />
                      Refresh
                    </button>
                    <span style={{ fontSize: 11, color: "hsl(var(--muted-foreground))" }}>{filteredPipeline.length} entries</span>
                  </>)}
                </div>

                {/* ── Console view ─────────────────────────────────────────── */}
                {logMode === "console" && (
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
                      const kind  = classifyLog(l.msg);
                      const color = kind === "error" ? "#f87171" : kind === "warn" ? "#fbbf24" : kind === "success" ? "#4ade80" : "#94a3b8";
                      const bg    = kind === "error" ? "rgba(248,113,113,0.04)" : kind === "warn" ? "rgba(251,191,36,0.04)" : "transparent";
                      const ts    = l.ts ? new Date(l.ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
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
                )}

                {/* ── Pipeline view ─────────────────────────────────────────── */}
                {logMode === "pipeline" && (
                  <div style={{
                    height: 560, overflowY: "auto", padding: "8px 0",
                    background: "hsl(220 15% 7%)",
                    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
                    fontSize: 12,
                  }}>
                    {filteredPipeline.length === 0 ? (
                      <div style={{ padding: 40, textAlign: "center", color: "#4a5568", fontSize: 13 }}>
                        {pipelineLoading ? "Loading pipeline logs…" : "No pipeline events yet. Run a query to see structured logs."}
                      </div>
                    ) : [...filteredPipeline].reverse().map((e, i) => {
                      const { ts, meta, summary, sessionShort } = fmtPipelineRow(e);
                      const isAlert = e.event === "alert";
                      const bg = isAlert ? "rgba(239,68,68,0.06)" : "transparent";
                      return (
                        <div key={i}
                          style={{ display: "flex", alignItems: "baseline", gap: 10, padding: "4px 16px", background: bg, borderLeft: isAlert ? "2px solid #ef4444" : "2px solid transparent", transition: "background 0.1s" }}
                          onMouseEnter={ev => (ev.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.03)"}
                          onMouseLeave={ev => (ev.currentTarget as HTMLDivElement).style.background = bg}
                        >
                          {/* timestamp */}
                          <span style={{ color: "#4a5568", flexShrink: 0, fontSize: 10, width: 60 }}>{ts}</span>
                          {/* event badge */}
                          <span style={{
                            flexShrink: 0, fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 4,
                            background: `${meta.color}20`, color: meta.color, border: `1px solid ${meta.color}35`,
                            width: 64, textAlign: "center",
                          }}>
                            {meta.icon} {meta.label}
                          </span>
                          {/* session chip */}
                          {sessionShort && (
                            <span style={{ flexShrink: 0, fontSize: 9, color: "#4a5568", fontFamily: "monospace" }}>
                              …{sessionShort}
                            </span>
                          )}
                          {/* summary */}
                          <span style={{ color: isAlert ? "#f87171" : "#94a3b8", wordBreak: "break-all", lineHeight: 1.7, flex: 1 }}>
                            {summary}
                          </span>
                        </div>
                      );
                    })}
                    <div ref={pipelineEndRef} />
                  </div>
                )}
              </div>

              {/* ── Pipeline stats cards (only in pipeline mode) ─────────── */}
              {logMode === "pipeline" && pipelineStats && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
                  {([
                    { label: "Total Queries",    value: (pipelineStats.event_counts?.["query_end"] ?? 0).toLocaleString(),      color: "#22c55e" },
                    { label: "Judge Pass Rate",  value: pipelineStats.judge_pass_rate != null ? `${(pipelineStats.judge_pass_rate * 100).toFixed(1)}%` : "—", color: "#a855f7" },
                    { label: "Judge Passes",     value: (pipelineStats.judge_pass ?? 0).toLocaleString(),                        color: "#3b82f6" },
                    { label: "Judge Failures",   value: (pipelineStats.judge_fail ?? 0).toLocaleString(),                        color: "#ef4444" },
                    { label: "Routing Events",   value: (pipelineStats.event_counts?.["routing"]   ?? 0).toLocaleString(),      color: "#f59e0b" },
                    { label: "Ingestion Events", value: (pipelineStats.event_counts?.["ingestion"] ?? 0).toLocaleString(),      color: "#06b6d4" },
                    { label: "Alerts Fired",     value: (pipelineStats.event_counts?.["alert"]     ?? 0).toLocaleString(),      color: "#ef4444" },
                  ]).map(s => (
                    <div key={s.label} style={{ ...S.card, padding: "14px 16px" }}>
                      <div style={{ fontSize: 20, fontWeight: 800, color: s.color, lineHeight: 1 }}>{s.value}</div>
                      <div style={{ fontSize: 10, color: "hsl(var(--muted-foreground))", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{s.label}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })()}

        {/* ══════════════════════ SETTINGS TAB ══════════════════════ */}
        {tab === "settings" && (
          <div style={{ maxWidth: 480 }}>
            <div style={S.card}>
              <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", gap: 10 }}>
                {/* Show admin's own avatar in settings */}
                {adminAvatar ? (
                  <img src={adminAvatar} referrerPolicy="no-referrer" alt={currentUser.username}
                    style={{ width: 36, height: 36, borderRadius: "50%", objectFit: "cover", border: "2px solid #7c3aed44" }} />
                ) : (
                  <div style={{
                    width: 36, height: 36, borderRadius: "50%",
                    background: "linear-gradient(135deg,#7c3aed,#4f46e5)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, fontWeight: 700, color: "#fff",
                  }}>
                    {currentUser.username[0]?.toUpperCase()}
                  </div>
                )}
                <div>
                  <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>My Admin Credentials</h3>
                  <div style={{ fontSize: 11, color: "hsl(var(--muted-foreground))", marginTop: 1 }}>{currentUser.email}</div>
                </div>
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
                  <label style={{ fontSize: 12, fontWeight: 600, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 6 }}>
                    New Password <span style={{ fontWeight: 400 }}>(leave blank to keep)</span>
                  </label>
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
                {selfMsg && (
                  <div style={{ fontSize: 12, color: selfMsg.includes("fail") ? "#ef4444" : "#22c55e", padding: "8px 12px", borderRadius: 8, background: selfMsg.includes("fail") ? "#ef444415" : "#22c55e15" }}>
                    {selfMsg}
                  </div>
                )}
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
            <div style={{ padding: "16px 20px", borderBottom: "1px solid hsl(var(--border))", display: "flex", alignItems: "center", gap: 10, justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                  background: editUser.avatar_url ? "transparent" : (editUser.role === "admin" ? "linear-gradient(135deg,#7c3aed,#4f46e5)" : "linear-gradient(135deg,#3b82f6,#06b6d4)"),
                  overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 700, color: "#fff",
                }}>
                  {editUser.avatar_url
                    ? <img src={editUser.avatar_url} referrerPolicy="no-referrer" alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    : editUser.username[0]?.toUpperCase()
                  }
                </div>
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "hsl(var(--foreground))" }}>Edit — {editUser.username}</h3>
              </div>
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

      {/* ══════════ EMAIL COMPOSE MODAL ══════════ */}
      {emailModal && (
        <div
          style={{ position: "fixed", inset: 0, zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.65)", backdropFilter: "blur(6px)" }}
          onClick={e => { if (e.target === e.currentTarget) setEmailModal(false); }}
        >
          <div style={{ width: 500, borderRadius: 20, overflow: "hidden", boxShadow: "0 32px 96px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.15)", background: "hsl(var(--card))" }}>

            {/* ── Branded hero header ── */}
            <div style={{
              position: "relative", padding: "28px 28px 24px",
              background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
              borderBottom: "1px solid rgba(59,130,246,0.2)",
              overflow: "hidden",
            }}>
              {/* radial glow blobs */}
              <div style={{ position: "absolute", top: -30, left: -30, width: 160, height: 160, borderRadius: "50%", background: "radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%)", pointerEvents: "none" }} />
              <div style={{ position: "absolute", bottom: -40, right: -20, width: 180, height: 180, borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.14) 0%, transparent 70%)", pointerEvents: "none" }} />

              {/* close */}
              <button onClick={() => setEmailModal(false)} style={{ position: "absolute", top: 14, right: 14, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, cursor: "pointer", color: "rgba(255,255,255,0.6)", padding: "4px 6px", display: "flex", alignItems: "center", justifyContent: "center", transition: "background 0.15s" }}
                onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.13)")}
                onMouseLeave={e => (e.currentTarget.style.background = "rgba(255,255,255,0.07)")}
              >
                <X size={14} />
              </button>

              {/* Logo + title */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, position: "relative" }}>
                <div style={{ width: 44, height: 44, borderRadius: 12, background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.3)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  {/* Inline favicon SVG */}
                  <svg viewBox="0 0 329.18 372.02" xmlns="http://www.w3.org/2000/svg" style={{ width: 24, height: 24 }}>
                    <defs>
                      <linearGradient id="emailLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style={{ stopColor: "#60a5fa", stopOpacity: 1 }} />
                        <stop offset="50%" style={{ stopColor: "#3b82f6", stopOpacity: 1 }} />
                        <stop offset="100%" style={{ stopColor: "#2563eb", stopOpacity: 1 }} />
                      </linearGradient>
                    </defs>
                    <path fill="url(#emailLogoGrad)" fillRule="evenodd" d="M111,242.83c-.01-24.57-.04-49.15-.02-73.72,0-5.04,2.6-8.44,7.87-10.31,16.79-5.96,33.58-11.91,50.4-17.78,33.06-11.55,65.93-23.65,99.1-34.89,16.3-5.52,32.43-11.51,48.66-17.21,7.6-2.67,12.15.57,12.15,8.59.02,49.05,0,98.09,0,147.14-1.7-.31-1.06-1.68-1.2-2.68-.1-.76.24-1.79-.83-2.05-1.95-.48-3.99-1.33-5.9-1.15-6.13.59-12.16-.22-18.21-.84-.74-.08-1.77-.33-1.6-1.04.54-2.29-.92-1.96-2.24-2.01-1.56-.05-2.79.15-3.01,2.13-.13,1.14-.84,2.21-2.26,1.85-6.06-1.53-12.15-.84-18.24-.32-.82.07-1.55-.37-2.3-.55-4.2-.99-8.13.84-12.19,1.41-.24.03-.52.37-.62.63-1.09,2.74-1.96,1.37-2.99-.08-.76-1.06-1.7-1.69-3.16-2.04-4.79-1.12-9.5,3.77-14.23.07-2.59,1.22-5.17-.1-7.76-.08-9.81.05-19.51,2.06-29.43.88-3.45-.41-7.2.48-10.75-.49-1.62-.45-3.4-.76-4.8-.24-4.3,1.58-8.56.68-12.88.58-4.26-.1-8.52-.14-12.84.35-5.22.6-10.69,1.5-15.89-.94-1.01-.48-2.38-.15-3.57.33-.58.23-1.4.49-1.75-.49.15-.16.25-.38.41-.42,1.71-.52,2.48-1.65,2.33-3.43-.05-.62-.04-1.38-.81-1.42-1.42-.07-1.95,1.3-2.88,2.03-.16.12-.27.37-.44.41-2.29.59-2.84,4.03-5.06,3.86-2.38-.18-5.14.49-7.14-.75-3.48-2.14-6.15-.05-9.06.96-1.69.58-1.55,1.63-.27,2.68,1.04.86,2.14.94,3.4.5.56-.19,1.26-.06,1.38.76.06.44-.2.88-.64.86-1.86-.08-4.32.59-5.4-.79-1.97-2.52-3.45-2.33-5.35-.29,1.9-2.04,3.38-2.22,5.35.29,1.08,1.38,3.54.71,5.4.79.44.02.7-.42.64-.86-.12-.82-.82-.95-1.38-.76-1.26.43-2.36.36-3.4-.5-1.27-1.05-1.42-2.1.27-2.68,2.91-1,5.58-3.1,9.06-.96,2.01,1.23,4.76.56,7.14.75,2.22.17,2.77-3.26,5.06-3.86.17-.04.28-.29.44-.41.93-.73,1.46-2.1,2.88-2.03.77.04.76.8.81,1.42.16,1.78-.62,2.91-2.33,3.43-.16.05-.26.27-.41.42.35.99,1.17.73,1.75.49,1.19-.47,2.56-.8,3.57-.33,5.19,2.44,10.67,1.54,15.89.94,4.32-.49,8.57-.45,12.84-.35,4.32.1,8.58,1,12.88-.58,1.39-.51,3.18-.2,4.8.24,3.55.98,7.3.09,10.75.49,9.92,1.18,19.63-.84,29.43-.88,2.59-.01,5.17,1.3,7.76.08,4.73,3.7,9.44-1.19,14.23-.07,1.47.34,2.41.98,3.16,2.04,1.03,1.44,1.9,2.81,2.99.08.1-.26.38-.6.62-.63,4.06-.57,7.99-2.39,12.19-1.41.76.18,1.48.62,2.3.55,6.09-.52,12.18-1.2,18.24.32,1.42.36,2.13-.71,2.26-1.85.22-1.98,1.45-2.18,3.01-2.13,1.32.04,2.78-.28,2.24,2.01-.17.71.86.97,1.6,1.04,6.05.62,12.07,1.42,18.21.84,1.92-.18,3.95.66,5.9,1.15,1.07.27.72,1.29.83,2.05.14,1.01-.5,2.37,1.2,2.68,0,15.43.02,30.87,0,46.3,0,5.09-2.08,8.03-6.9,9.91-10.26,3.98-20.75,7.32-31.16,10.88-14.41,4.94-28.75,10.07-43.1,15.17-15.98,5.68-31.92,11.44-47.9,17.12-19.08,6.78-38.18,13.53-57.27,20.27-6.87,2.42-13.75,4.83-20.66,7.13-5.89,1.96-10.52-1.14-11.2-7.36-.05-.5-.02-1.01-.02-1.52,0-32.39.02-64.78.03-97.17,1.69-1.66,1.69-4.45,0-6.11,0-2.64,0-5.27,0-7.91.31-.16.83-.27.88-.49.15-.63-.44-.65-.88-.75,0-2.43,0-4.86,0-7.3,1.69,1.67,1.69,4.45,0,6.11,0-2.04,0-4.07,0-6.11,.44.1,1.02.11.88.75-.05.22-.57.33-.88.49,0-.41,0-.82,0-1.24ZM.03,169.67c0-25.09-.02-50.19,0-75.28,0-5.65,2.97-8.72,8.46-10.17,4.66-1.23,9.11-3.27,13.67-4.9,12.4-4.43,24.79-8.87,37.21-13.25,29.03-10.23,58.07-20.41,87.11-30.63,17.37-6.12,34.73-12.26,52.09-18.41,3.9-1.38,7.62-3.37,11.89-3.4,3.68-.03,6.35,1.87,6.99,5.43.39,2.16.84,4.34.83,6.6-.06,19.1.06,38.2-.08,57.3-.06,7.91-2.05,9.73-8.75,12.24-8.9,3.34-18.02,6.11-26.98,9.29-12.3,4.37-24.47,9.15-36.85,13.28-7.18,2.4-14.32,4.92-21.47,7.41-13.27,4.62-26.47,9.45-39.71,14.18-3.11,1.11-4.73,3.03-4.65,6.6.1,4.24-.33,8.48-.53,12.73-1.49.1-2.11-.58-1.68-2.03.71-2.35-.53-2.86-2.52-2.86-4.12,0-8.18,1-12.35.41-2.77-.39-5.64-.6-8.41-.31-7.15.75-14.31.32-21.46.56-1.67.06-3.7-2.14-5.05.69-.19.4-.75.17-.77-.35-.09-1.68-1.36-1.04-2-.83-1.5.49-3,.49-4.51.45-2.51-.07-4.79.29-6.67,2.25-1.18,1.23-2.61,1.32-3.39-.51-.58-1.35-1.43-1.12-2.47-.84-4.19,1.13-5.79,3.46-6.25,7.78-.23,2.19.47,4.82-1.71,6.57,2.18-1.75,1.48-4.38,1.71-6.57.46-4.32,2.05-6.65,6.25-7.78,1.04-.28,1.89-.52,2.47.84.78,1.83,2.22,1.74,3.39.51,1.88-1.96,4.16-2.32,6.67-2.25,1.51.04,3.01.05,4.51-.45.64-.21,1.91-.84,2,.83.03.52.58.75.77.35,1.35-2.83,3.38-.64,5.05-.69,7.15-.24,14.31.19,21.46-.56,2.77-.29,5.64-.08,8.41.31,4.17.59,8.23-.4,12.35-.41,1.99,0,3.23.51,2.52,2.86-.44,1.45.18,2.12,1.68,2.03,1.22,34.51.32,69.02.55,103.54.02,2.41-.16,4.88-.68,7.23-.6,2.66-2.18,4.69-4.98,5.68-16.15,5.67-32.27,11.41-48.42,17.11-5.44,1.92-10.85,3.94-16.37,5.6-4.44,1.34-8.93-2-9.35-6.59-.05-.5-.02-1.02-.02-1.52,0-40.02.01-80.04.02-120.06Z" />
                  </svg>
                </div>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: "#fff", letterSpacing: "-0.01em" }}>
                    {emailTarget ? `Message to ${emailTarget.username}` : "Broadcast Message"}
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(148,163,184,0.9)", marginTop: 2 }}>
                    {emailTarget
                      ? `Sending to ${emailTarget.email}`
                      : `${users.length} recipient${users.length !== 1 ? "s" : ""} · All users`}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Form body ── */}
            <div style={{ padding: "22px 24px", display: "flex", flexDirection: "column", gap: 16 }}>

              {/* To chip */}
              <div>
                <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 7, letterSpacing: "0.05em", textTransform: "uppercase" }}>To</label>
                <div style={{
                  display: "flex", alignItems: "center", gap: 8, padding: "8px 12px",
                  borderRadius: 9, border: "1px solid hsl(var(--border))",
                  background: "hsl(var(--muted)/0.3)",
                }}>
                  {emailTarget ? (
                    <>
                      {emailTarget.avatar_url
                        ? <img src={emailTarget.avatar_url} referrerPolicy="no-referrer" alt="" style={{ width: 22, height: 22, borderRadius: "50%", objectFit: "cover", border: "1.5px solid #3b82f640" }} />
                        : <div style={{ width: 22, height: 22, borderRadius: "50%", background: "linear-gradient(135deg,#3b82f6,#06b6d4)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "#fff", flexShrink: 0 }}>{emailTarget.username[0]?.toUpperCase()}</div>
                      }
                      <span style={{ fontSize: 13, fontWeight: 600, color: "hsl(var(--foreground))" }}>{emailTarget.username}</span>
                      <span style={{ fontSize: 12, color: "hsl(var(--muted-foreground))" }}>·</span>
                      <span style={{ fontSize: 12, color: "hsl(var(--muted-foreground))" }}>{emailTarget.email}</span>
                    </>
                  ) : (
                    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                      <div style={{ width: 22, height: 22, borderRadius: "50%", background: "linear-gradient(135deg,#3b82f6,#7c3aed)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, flexShrink: 0 }}>
                        <Users size={11} color="#fff" />
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "hsl(var(--foreground))" }}>All Users</span>
                      <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 99, background: "#3b82f618", border: "1px solid #3b82f630", color: "#3b82f6", fontWeight: 700 }}>{users.length}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Subject */}
              <div>
                <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 7, letterSpacing: "0.05em", textTransform: "uppercase" }}>Subject</label>
                <input
                  value={emailSubject}
                  onChange={e => setEmailSubject(e.target.value)}
                  placeholder="Enter subject…"
                  style={{
                    ...S.input,
                    fontSize: 13, fontWeight: 500,
                    background: "hsl(var(--muted)/0.35)",
                    transition: "border-color 0.15s",
                  }}
                  onFocus={e => (e.currentTarget.style.borderColor = "#3b82f6")}
                  onBlur={e => (e.currentTarget.style.borderColor = "hsl(var(--border))")}
                />
              </div>

              {/* Body */}
              <div>
                <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", display: "block", marginBottom: 7, letterSpacing: "0.05em", textTransform: "uppercase" }}>Message</label>
                <textarea
                  value={emailBody}
                  onChange={e => setEmailBody(e.target.value)}
                  placeholder="Write your message here…"
                  rows={5}
                  style={{
                    ...S.input, resize: "vertical", lineHeight: 1.7, fontFamily: "inherit",
                    fontSize: 13, background: "hsl(var(--muted)/0.35)", transition: "border-color 0.15s",
                  }}
                  onFocus={e => (e.currentTarget.style.borderColor = "#3b82f6")}
                  onBlur={e => (e.currentTarget.style.borderColor = "hsl(var(--border))")}
                />
              </div>

              {/* Footer row */}
              <div style={{ display: "flex", gap: 8, alignItems: "center", paddingTop: 4 }}>
                <button
                  onClick={sendEmail}
                  disabled={emailSending || !emailSubject.trim() || !emailBody.trim()}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 7,
                    padding: "9px 18px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                    cursor: (emailSending || !emailSubject.trim() || !emailBody.trim()) ? "not-allowed" : "pointer",
                    border: "none",
                    background: (emailSending || !emailSubject.trim() || !emailBody.trim())
                      ? "rgba(59,130,246,0.35)"
                      : "linear-gradient(135deg, #3b82f6, #2563eb)",
                    color: "#fff",
                    boxShadow: (emailSending || !emailSubject.trim() || !emailBody.trim()) ? "none" : "0 4px 14px rgba(59,130,246,0.35)",
                    transition: "all 0.15s",
                  }}
                >
                  {emailSending
                    ? <RefreshCw size={13} style={{ animation: "spin 0.8s linear infinite" }} />
                    : <Send size={13} />
                  }
                  {emailSending ? "Sending…" : "Send Message"}
                </button>
                <button onClick={() => setEmailModal(false)} style={{ ...S.btn("ghost"), padding: "9px 14px", fontSize: 13 }}>
                  <X size={13} /> Cancel
                </button>
                {emailResult && (
                  <div style={{
                    marginLeft: 4, fontSize: 12, fontWeight: 600,
                    color: emailResult.startsWith("✅") ? "#22c55e" : "#ef4444",
                    display: "flex", alignItems: "center", gap: 5,
                  }}>
                    {emailResult}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        @keyframes kpiChartFadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
      `}</style>
    </div>
  );
}