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
import AuthModal from "@/components/AuthModal";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface AuthUser {
  token:    string;
  user_id:  string;
  username: string;
  email:    string;
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

  // Rehydrate from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem("bimlo_auth");
      if (stored) {
        const parsed: AuthUser = JSON.parse(stored);
        if (parsed.token && parsed.user_id) {
          setCurrentUserState(parsed);
        }
      }
    } catch {
      localStorage.removeItem("bimlo_auth");
    }
  }, []);

  const setCurrentUser = useCallback((u: AuthUser | null) => {
    setCurrentUserState(u);
    if (u) {
      localStorage.setItem("bimlo_auth", JSON.stringify(u));
    } else {
      localStorage.removeItem("bimlo_auth");
    }
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
    setCurrentUser(user);
    setModalOpen(false); // belt-and-suspenders in case onClose wasn't called
    if (pendingActionRef.current) {
      const action = pendingActionRef.current;
      pendingActionRef.current = null;
      setTimeout(action, 150); // wait for modal exit animation
    }
  }, [setCurrentUser]);

  const logout = useCallback(async () => {
    if (currentUser?.token) {
      try {
        await fetch("/auth/logout", {
          method:  "POST",
          headers: { Authorization: `Bearer ${currentUser.token}` },
        });
      } catch { /* ignore */ }
    }
    setCurrentUser(null);
  }, [currentUser, setCurrentUser]);

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