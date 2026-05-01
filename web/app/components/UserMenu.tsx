"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "./AuthProvider";

/**
 * Small avatar + dropdown shown in the topbar. Renders nothing if the
 * user is unauthed (the topbar can show login/register buttons in that
 * case instead).
 */
export function UserMenu() {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  if (!user) return null;

  const initials = user.fullName
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("") || "DR";

  return (
    <div className="user-menu" ref={ref}>
      <button
        className="user-menu__trigger"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="user-menu__avatar">{initials}</span>
        <span className="user-menu__name">{user.fullName}</span>
      </button>
      {open && (
        <div className="user-menu__panel" role="menu">
          <div className="user-menu__head">
            <div className="user-menu__head-name">{user.fullName}</div>
            <div className="user-menu__head-email">{user.email}</div>
            {user.specialty && (
              <div className="user-menu__head-meta">{user.specialty}</div>
            )}
          </div>
          <Link
            href="/profile"
            className="user-menu__item"
            onClick={() => setOpen(false)}
          >
            Profile
          </Link>
          <Link
            href="/app"
            className="user-menu__item"
            onClick={() => setOpen(false)}
          >
            Diagnostic tool
          </Link>
          <Link
            href="/insights"
            className="user-menu__item"
            onClick={() => setOpen(false)}
          >
            Insights
          </Link>
          <div className="user-menu__sep" />
          <button
            className="user-menu__item user-menu__item--danger"
            onClick={() => {
              setOpen(false);
              logout();
              router.replace("/");
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
