/**
 * AuthContext.tsx
 *
 * Global auth state for BIMLO Copilot.
 * Wraps the app and exposes:
 *   - currentUser: { token, user_id, username, email } | null
 *   - showAuthModal(pendingAction?)  — open the login/signup popup
 *   - hideAuthModal()
 *   - logout()
 *
 * The modal shows on demand (when user sends first message, visits /call or /news).
 * It NEVER redirects — the user's pending action (e.g. the message they tried to send)
 * is preserved and fired immediately after login.
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from "react";
import api from "@/services/api";
import AuthModal from "@/components/AuthModal";

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface AuthUser {
  token:         string;
  user_id:       string;
  username:      string;
  email:         string;
  avatar_url?:   string;
  display_name?: string;
  role?:         string;  // "user" | "admin"
}

export type PendingAction = (() => void) | null;

interface AuthContextValue {
  currentUser:      AuthUser | null;
  isLoggedIn:       boolean;
  showAuthModal:    (onSuccess?: PendingAction) => void;
  hideAuthModal:    () => void;
  logout:           () => void;
  setCurrentUser:   (u: AuthUser | null) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context
// ─────────────────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue>({
  currentUser:      null,
  isLoggedIn:       false,
  showAuthModal:    () => {},
  hideAuthModal:    () => {},
  logout:           () => {},
  setCurrentUser:   () => {},
});

export const useAuth = () => useContext(AuthContext);

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentUser, setCurrentUserState] = useState<AuthUser | null>(null);
  const [modalOpen, setModalOpen]           = useState(false);
  const pendingActionRef                    = useRef<PendingAction>(null);

  // Rehydrate via /auth/me (HttpOnly cookie-based session)
  useEffect(() => {
      fetch(`${API_BASE_URL}/auth/me`, { credentials: "include" })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        if (data.user_id) {
          // Try to get a fresh access token
          fetch(`${API_BASE_URL}/auth/refresh`, { method: "POST", credentials: "include" })
            .then(r => r.ok ? r.json() : null)
            .then(refresh => {
              if (refresh?.access_token) {
                api.setToken(refresh.access_token);
              }
            })
            .catch(() => {});
          setCurrentUserState({
            token: "",
            user_id: data.user_id,
            username: data.username,
            email: data.email,
            role: data.role,
            avatar_url: data.avatar_url || "",
          });
        }
      })
      .catch(() => {});
  }, []);

  const setCurrentUser = useCallback((u: AuthUser | null) => {
    setCurrentUserState(u);
  }, []);

  const showAuthModal = useCallback((onSuccess: PendingAction = null) => {
    pendingActionRef.current = onSuccess;
    setModalOpen(true);
  }, []);

  const hideAuthModal = useCallback(() => {
    setModalOpen(false);
    pendingActionRef.current = null;
  }, []);

  const handleAuthSuccess = useCallback((user: AuthUser) => {
    api.setToken(user.token);
    setCurrentUser(user);
    setModalOpen(false);
    if (pendingActionRef.current) {
      const action = pendingActionRef.current;
      pendingActionRef.current = null;
      setTimeout(action, 150);
    }
  }, [setCurrentUser]);

  // ── Heartbeat: ping /auth/heartbeat every 60s while logged in ──────────────
  useEffect(() => {
    if (!currentUser?.user_id) return;
    const ping = () =>
      fetch(`${API_BASE_URL}/auth/heartbeat`, {
        method: "POST",
        credentials: "include",
      }).catch(() => {});
    ping();
    const id = setInterval(ping, 60_000);
    return () => clearInterval(id);
  }, [currentUser?.user_id]);

  const logout = useCallback(async () => {
    try {
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch { /* ignore */ }
    api.setToken(null);
    setCurrentUser(null);
  }, [setCurrentUser]);

  return (
    <AuthContext.Provider
      value={{
        currentUser,
        isLoggedIn: !!currentUser,
        showAuthModal,
        hideAuthModal,
        logout,
        setCurrentUser,
      }}
    >
      {children}

      <AuthModal
        open={modalOpen}
        onClose={hideAuthModal}
        onSuccess={handleAuthSuccess}
      />
    </AuthContext.Provider>
  );
};