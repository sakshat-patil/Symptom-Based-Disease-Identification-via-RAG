"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "./AuthProvider";

/**
 * Wrap any page that requires sign-in. While we boot the auth context
 * we render a tiny placeholder; once we know the user is unauthed we
 * redirect to /login. This is a UX guard, not a security boundary.
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  if (loading) {
    return (
      <div className="auth-loading">
        <div className="auth-loading__spinner" />
      </div>
    );
  }
  if (!user) {
    return null;
  }
  return <>{children}</>;
}
