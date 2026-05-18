import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2, Eye, EyeOff, LogIn, UserPlus } from "lucide-react";
import { useTranslation } from "react-i18next";

declare global { interface Window { google?: any; } }

export interface AuthUser {
  token:         string;
  user_id:       string;
  username:      string;
  email:         string;
  avatar_url?:   string;
  display_name?: string;
}

interface Props {
  open:      boolean;
  onClose:   () => void;
  onSuccess: (user: AuthUser) => void;
}

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || "";

const RATE_LIMIT_KEY = "bimlo_rate_limit_until";

async function apiPost(path: string, body: object): Promise<{ data?: AuthUser; error?: string; rateLimited?: boolean }> {
  try {
    const res = await fetch(`${BASE}${path}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body:    JSON.stringify(body),
    });
    const json = await res.json();
    if (!res.ok) {
      if (res.status === 429) {
        const expiresAt = Date.now() + 60_000; // 1 minute from now
        localStorage.setItem(RATE_LIMIT_KEY, String(expiresAt));
        return { error: "Too many attempts.", rateLimited: true };
      }
      return { error: json.detail || "Something went wrong" };
    }
    // Note: error strings are not translated here — they come from the backend
    return { data: json as AuthUser };
  } catch {
    return { error: "Network error — is the server running?" };
  }
}

const Input: React.FC<{
  label:        string;
  type?:        string;
  value:        string;
  onChange:     (v: string) => void;
  placeholder?: string;
  autoFocus?:   boolean;
  suffix?:      React.ReactNode;
}> = ({ label, type = "text", value, onChange, placeholder, autoFocus, suffix }) => (
  <div className="flex flex-col gap-1">
    <label className="text-[11px] font-medium tracking-wide uppercase" style={{ color: "#64748b" }}>
      {label}
    </label>
    <div className="relative">
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        autoFocus={autoFocus}
        className="w-full text-sm px-3 py-2 rounded-lg outline-none transition-all"
        style={{
          background:   "rgba(100,116,139,0.10)",
          border:       "1px solid rgba(100,116,139,0.25)",
          color:        "inherit",
          paddingInlineEnd: suffix ? "2.5rem" : undefined,
        }}
        onFocus={e => {
          e.currentTarget.style.borderColor = "#60a5fa";
          e.currentTarget.style.boxShadow   = "0 0 0 2px rgba(96,165,250,0.15)";
        }}
        onBlur={e => {
          e.currentTarget.style.borderColor = "rgba(100,116,139,0.25)";
          e.currentTarget.style.boxShadow   = "";
        }}
      />
      {suffix && (
        <div className="absolute inset-y-0 end-0 flex items-center pe-3">{suffix}</div>
      )}
    </div>
  </div>
);

const GoogleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
    <path fill="none" d="M0 0h48v48H0z"/>
  </svg>
);

export default function AuthModal({ open, onClose, onSuccess }: Props) {
  const { t } = useTranslation();
  const [tab,      setTab]      = useState<"login" | "signup">("login");
  const [email,    setEmail]    = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPw,   setShowPw]   = useState(false);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [rateLimitSecs, setRateLimitSecs] = useState(0);

  // Restore rate limit countdown from localStorage on mount/open
  useEffect(() => {
    if (!open) return;
    const stored = localStorage.getItem(RATE_LIMIT_KEY);
    if (stored) {
      const remaining = Math.ceil((Number(stored) - Date.now()) / 1000);
      if (remaining > 0) {
        setRateLimitSecs(remaining);
      } else {
        localStorage.removeItem(RATE_LIMIT_KEY);
        setRateLimitSecs(0);
      }
    }
  }, [open]);

  // Countdown timer — persists across refreshes via localStorage
  useEffect(() => {
    if (rateLimitSecs <= 0) return;
    const interval = setInterval(() => {
      setRateLimitSecs(prev => {
        if (prev <= 1) {
          localStorage.removeItem(RATE_LIMIT_KEY);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [rateLimitSecs]);

  useEffect(() => {
    if (open) {
      setEmail(""); setUsername(""); setPassword("");
      setError(null); setLoading(false); setShowPw(false);
    }
  }, [open, tab]);

  // Intentionally no body/html overflow toggling —
  // the fixed backdrop already blocks interaction,
  // and toggling overflow causes scrollbar jitter.


  const handleGoogleClick = () => {
    if (!GOOGLE_CLIENT_ID) { setError(t("auth.google_not_configured")); return; }

    const loadAndOpen = () => {
      const client = window.google.accounts.oauth2.initTokenClient({
        client_id: GOOGLE_CLIENT_ID,
        scope: "openid email profile",
        callback: async (tokenResponse: any) => {
          if (tokenResponse.error) { setError(t("auth.google_signin_failed")); return; }
          // Exchange access token for user info, then send to backend as credential
          setLoading(true);
          setError(null);
          try {
            // Get user info from Google
            const infoRes = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
              headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
            });
            const info = await infoRes.json();
            console.log("[BIMLO] Google userinfo full response:", info);
            console.log("[BIMLO] picture field:", info.picture);
            // Send to our backend — we'll pass the access token, backend verifies via userinfo
            const { data, error: err } = await apiPost("/auth/google-token", {
              access_token: tokenResponse.access_token,
              email: info.email,
              name: info.name,
              sub: info.sub,
              picture: info.picture,
            });
            console.log("[BIMLO] /auth/google-token response:", data, "error:", err);
            if (err) { setError(err); }
            else if (data) {
              const userWithAvatar = { ...data, avatar_url: info.picture };
              console.log("[BIMLO] calling onSuccess with:", userWithAvatar);
              onClose();
              setTimeout(() => onSuccess(userWithAvatar), 50);
            }
          } catch { setError(t("auth.google_signin_failed_try_again")); }
          finally { setLoading(false); }
        },
      });
      client.requestAccessToken({ prompt: "select_account" });
    };

    if (window.google?.accounts?.oauth2) {
      loadAndOpen();
    } else {
      const script = document.createElement("script");
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.onload = loadAndOpen;
      document.head.appendChild(script);
    }
  };

  const handleSubmit = async () => {
    setError(null);
    if (!email.trim() || !password.trim()) { setError(t("validation.fill_all_fields")); return; }
    if (tab === "signup" && !username.trim()) { setError(t("validation.choose_username")); return; }
    setLoading(true);
    const path = tab === "login" ? "/auth/login" : "/auth/signup";
    const body = tab === "login"
      ? { email: email.trim(), password }
      : { email: email.trim(), username: username.trim(), password };
    const { data, error: err, rateLimited } = await apiPost(path, body);
    if (err) {
      setLoading(false);
      if (rateLimited) {
        setRateLimitSecs(60);
        setError(null);
      } else {
        setError(err);
      }
      return;
    }
    if (data) {
      // Dismiss first so the modal is gone before the pending action fires
      onClose();
      setTimeout(() => onSuccess(data), 50);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSubmit();
    if (e.key === "Escape") onClose();
  };

  const isDark     = document.documentElement.classList.contains("dark");
  const overlayBg  = isDark ? "rgba(2,6,23,0.75)"    : "rgba(15,23,42,0.45)";
  const cardBg     = isDark ? "rgba(15,23,42,0.97)"   : "rgba(255,255,255,0.98)";
  const cardBorder = isDark ? "rgba(51,65,85,0.7)"    : "rgba(203,213,225,0.8)";
  const mutedFg    = isDark ? "#64748b"                : "#94a3b8";
  const fg         = isDark ? "#e2e8f0"                : "#1e293b";
  const dividerClr = isDark ? "rgba(100,116,139,0.2)" : "rgba(203,213,225,0.6)";

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
            style={{
              position: "fixed", inset: 0, zIndex: 9998,
              background: overlayBg,
              backdropFilter: "blur(6px)",
              WebkitBackdropFilter: "blur(6px)",
            }}
          />

          {/* Centering shell */}
          <div style={{
            position: "fixed", inset: 0, zIndex: 9999,
            display: "flex", alignItems: "center", justifyContent: "center",
            pointerEvents: "none",
          }}>
            <motion.div
              key="card"
              initial={{ opacity: 0, scale: 0.93, y: 20 }}
              animate={{ opacity: 1, scale: 1,    y: 0  }}
              exit={{   opacity: 0, scale: 0.93, y: 20  }}
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
              onKeyDown={handleKey}
              onClick={e => e.stopPropagation()}
              style={{
                pointerEvents: "all",
                width:         "min(92vw, 368px)",
                background:    cardBg,
                border:        `1px solid ${cardBorder}`,
                borderRadius:  18,
                boxShadow:     "0 28px 72px rgba(0,0,0,0.30), 0 0 0 1px rgba(96,165,250,0.07)",
                padding:       26,
                display:       "flex",
                flexDirection: "column",
                gap:           18,
              }}
            >
              {/* Header */}
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                <div>
                  <p style={{ fontSize: 11, color: mutedFg, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 2 }}>
                    BIMLO Copilot
                  </p>
                  <h2 style={{ fontSize: 19, fontWeight: 700, color: fg, margin: 0 }}>
                    {tab === "login" ? t("auth.welcome_back") : t("auth.create_account")}
                  </h2>
                </div>
                <button onClick={onClose} style={{ background: "transparent", border: "none", cursor: "pointer", color: mutedFg, padding: 4, borderRadius: 6, display: "flex", alignItems: "center" }}>
                  <X size={16} />
                </button>
              </div>

              {/* Tab switcher */}
              <div style={{ display: "flex", gap: 2, padding: 3, background: isDark ? "rgba(30,41,59,0.6)" : "rgba(241,245,249,1)", borderRadius: 10 }}>
                {(["login", "signup"] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => { setTab(mode); setError(null); }}
                    style={{
                      flex: 1, padding: "7px", border: "none", borderRadius: 8,
                      cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s",
                      background: tab === mode ? "#3b82f6" : "transparent",
                      color:      tab === mode ? "#fff"    : mutedFg,
                      boxShadow:  tab === mode ? "0 1px 6px rgba(59,130,246,0.3)" : "none",
                    }}
                  >
                    {mode === "login" ? t("auth.login") : t("auth.signup")}
                  </button>
                ))}
              </div>

              {/* Google Sign-In */}
              <button
                onClick={handleGoogleClick}
                style={{
                  width: "100%", display: "flex", alignItems: "center", justifyContent: "center",
                  gap: 10, padding: "10px", borderRadius: 10, cursor: "pointer",
                  fontSize: 13, fontWeight: 600,
                  background: isDark ? "rgba(255,255,255,0.06)" : "#fff",
                  border: `1px solid ${dividerClr}`,
                  color: fg, transition: "background 0.15s",
                  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.11)" : "#f8fafc"; }}
                onMouseLeave={e => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.06)" : "#fff"; }}
              >
                <GoogleIcon />
                {t("auth.google_login")}
              </button>

              {/* Divider */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ flex: 1, height: 1, background: dividerClr }} />
                <span style={{ fontSize: 11, color: mutedFg, whiteSpace: "nowrap" }}>{t("auth.or_with_email")}</span>
                <div style={{ flex: 1, height: 1, background: dividerClr }} />
              </div>

              {/* Form */}
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <Input label={t("auth.email")} type="email" value={email} onChange={setEmail} placeholder={t("auth.email_placeholder")} autoFocus />
                <AnimatePresence>
                  {tab === "signup" && (
                    <motion.div
                      key="username-field"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{   opacity: 0, height: 0 }}
                      transition={{ duration: 0.18 }}
                      style={{ overflow: "hidden" }}
                    >
                      <Input label={t("auth.username")} value={username} onChange={setUsername} placeholder={t("auth.username_placeholder")} />
                    </motion.div>
                  )}
                </AnimatePresence>
                <Input
                  label={t("auth.password")}
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={setPassword}
                  placeholder={tab === "signup" ? t("auth.password_placeholder_signup") : t("auth.password_placeholder_login")}
                  suffix={
                    <button type="button" onClick={() => setShowPw(v => !v)}
                      style={{ background: "transparent", border: "none", cursor: "pointer", color: mutedFg, display: "flex" }}>
                      {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  }
                />
              </div>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    style={{ fontSize: 12, color: "#f87171", background: "rgba(248,113,113,0.10)", border: "1px solid rgba(248,113,113,0.25)", borderRadius: 8, padding: "8px 12px" }}
                  >
                    {error}
                  </motion.div>
                )}
                {rateLimitSecs > 0 && (
                  <motion.div
                    key="rate-limit"
                    initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    style={{ fontSize: 12, color: "#f59e0b", background: "rgba(245,158,11,0.10)", border: "1px solid rgba(245,158,11,0.25)", borderRadius: 8, padding: "8px 12px" }}
                  >
                    Too many attempts. Try again in {rateLimitSecs}s
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={loading || rateLimitSecs > 0}
                style={{
                  width: "100%", padding: "11px", borderRadius: 10, border: "none",
                  cursor: (loading || rateLimitSecs > 0) ? "not-allowed" : "pointer",
                  background: (loading || rateLimitSecs > 0) ? "rgba(59,130,246,0.5)" : "#3b82f6",
                  color: "#fff", fontSize: 14, fontWeight: 600,
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "background 0.15s",
                  boxShadow: (loading || rateLimitSecs > 0) ? "none" : "0 4px 14px rgba(59,130,246,0.35)",
                }}
                onMouseEnter={e => { if (!loading && rateLimitSecs <= 0) e.currentTarget.style.background = "#2563eb"; }}
                onMouseLeave={e => { if (!loading && rateLimitSecs <= 0) e.currentTarget.style.background = "#3b82f6"; }}
              >
                {loading
                  ? <Loader2 size={15} className="animate-spin" />
                  : tab === "login"
                    ? <><LogIn size={15} /> {t("auth.login")}</>
                    : <><UserPlus size={15} /> {t("auth.create_account")}</>
                }
              </button>

              {/* Guest */}
              <p style={{ textAlign: "center", fontSize: 12, color: mutedFg, margin: 0 }}>
                or{" "}
                <button onClick={onClose} style={{ background: "transparent", border: "none", cursor: "pointer", color: mutedFg, textDecoration: "underline", fontSize: 12, padding: 0 }}>
                  {t("auth.continue_as_guest")}
                </button>
                {" "}{t("auth.no_history_saved")}
              </p>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}