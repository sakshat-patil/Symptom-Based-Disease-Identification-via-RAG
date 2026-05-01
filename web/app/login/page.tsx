"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "../components/AuthProvider";

export default function LoginPage() {
  const { login, user, loading } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && user) router.replace("/app");
  }, [loading, user, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await login(email, password);
      router.replace("/app");
    } catch (e: any) {
      setErr(e.message ?? "Sign in failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <Link href="/" className="auth__back" aria-label="Back to landing">
        ← Back
      </Link>
      <div className="auth__card">
        <div className="auth__head">
          <div className="topbar__logo">R</div>
          <div>
            <h1 className="auth__title">Sign in</h1>
            <p className="auth__sub">
              Welcome back. Pick up where you left off.
            </p>
          </div>
        </div>
        <form className="auth__form" onSubmit={onSubmit}>
          <label className="auth__label">
            Email
            <input
              type="email"
              required
              autoFocus
              autoComplete="email"
              className="input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="dr.kim@example.com"
            />
          </label>
          <label className="auth__label">
            Password
            <input
              type="password"
              required
              autoComplete="current-password"
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </label>
          {err && <div className="error-banner">{err}</div>}
          <button
            type="submit"
            className="btn btn--primary"
            disabled={busy}
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <div className="auth__foot">
          New here?{" "}
          <Link href="/register" className="auth__link">
            Create a doctor account
          </Link>
        </div>
        <div className="auth__fineprint">
          Local-only mock auth. Credentials live in your browser&rsquo;s
          storage and are not sent anywhere.
        </div>
      </div>
    </div>
  );
}
