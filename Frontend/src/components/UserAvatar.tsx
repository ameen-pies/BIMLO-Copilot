/**
 * UserAvatar.tsx
 *
 * Displays the user's Google profile picture if available,
 * falls back to a styled initials circle for email/password users.
 *
 * Usage:
 *   import UserAvatar from "@/components/UserAvatar";
 *
 *   <UserAvatar />                    — auto-reads from useAuth()
 *   <UserAvatar size={40} />          — custom size in px
 *   <UserAvatar showName />           — show username next to avatar
 *   <UserAvatar onClick={openMenu} /> — clickable (adds ring + cursor)
 */

import React, { useState } from "react";
import { useAuth } from "@/context/AuthContext";

// ─── color palette for initial avatars ───────────────────────────────────────
// Each user gets a stable color based on their username/email first char
const PALETTE = [
  { bg: "#dbeafe", text: "#1e40af" }, // blue
  { bg: "#dcfce7", text: "#166534" }, // green
  { bg: "#fce7f3", text: "#9d174d" }, // pink
  { bg: "#ede9fe", text: "#5b21b6" }, // purple
  { bg: "#ffedd5", text: "#9a3412" }, // orange
  { bg: "#cffafe", text: "#164e63" }, // cyan
  { bg: "#fef9c3", text: "#854d0e" }, // yellow
  { bg: "#fee2e2", text: "#991b1b" }, // red
];

function getColor(seed: string) {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = seed.charCodeAt(i) + ((hash << 5) - hash);
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

function getInitials(username: string, email: string): string {
  const name = username || email || "?";
  const parts = name.replace(/@.*/, "").split(/[\s._\-]+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

// ─── Props ────────────────────────────────────────────────────────────────────

interface UserAvatarProps {
  size?:      number;          // diameter in px (default: 36)
  showName?:  boolean;         // show username label beside avatar
  onClick?:   () => void;      // if provided, avatar becomes clickable
  className?: string;
  style?:     React.CSSProperties;
}

// ─── Component ────────────────────────────────────────────────────────────────

const UserAvatar: React.FC<UserAvatarProps> = ({
  size      = 36,
  showName  = false,
  onClick,
  className = "",
  style     = {},
}) => {
  const { currentUser } = useAuth();
  const [imgError, setImgError] = useState(false);

  if (!currentUser) return null;

  const { username, email, avatar_url } = currentUser;
  const initials = getInitials(username, email);
  const color    = getColor(username || email);
  const showImg  = !!avatar_url && !imgError;
  const clickable = !!onClick;

  const avatarStyle: React.CSSProperties = {
    width:          size,
    height:         size,
    minWidth:       size,
    borderRadius:   "50%",
    overflow:       "hidden",
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    fontSize:       Math.round(size * 0.38),
    fontWeight:     600,
    fontFamily:     "inherit",
    userSelect:     "none",
    cursor:         clickable ? "pointer" : "default",
    transition:     "box-shadow 0.15s, transform 0.12s",
    ...(showImg
      ? { background: "transparent" }
      : { background: color.bg, color: color.text }),
  };

  const wrapperStyle: React.CSSProperties = {
    display:     "inline-flex",
    alignItems:  "center",
    gap:         8,
    cursor:      clickable ? "pointer" : "default",
    ...style,
  };

  const handleMouseEnter = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!clickable) return;
    (e.currentTarget.firstElementChild as HTMLElement).style.boxShadow =
      "0 0 0 2px rgba(96,165,250,0.55)";
    (e.currentTarget.firstElementChild as HTMLElement).style.transform = "scale(1.06)";
  };

  const handleMouseLeave = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!clickable) return;
    (e.currentTarget.firstElementChild as HTMLElement).style.boxShadow = "";
    (e.currentTarget.firstElementChild as HTMLElement).style.transform  = "scale(1)";
  };

  return (
    <div
      style={wrapperStyle}
      className={className}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      title={username || email}
    >
      {/* Avatar circle */}
      <div style={avatarStyle}>
        {showImg ? (
          <img
            src={avatar_url}
            alt={username || email}
            referrerPolicy="no-referrer"
            onError={() => setImgError(true)}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <span aria-hidden="true">{initials}</span>
        )}
      </div>

      {/* Optional username label */}
      {showName && (
        <span
          style={{
            fontSize:   Math.max(13, Math.round(size * 0.38)),
            fontWeight: 500,
            color:      "inherit",
            whiteSpace: "nowrap",
            maxWidth:   120,
            overflow:   "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {username || email.split("@")[0]}
        </span>
      )}
    </div>
  );
};

export default UserAvatar;
