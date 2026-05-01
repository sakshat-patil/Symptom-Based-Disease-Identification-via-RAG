"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  DoctorProfile,
  getCurrentUser,
  logoutDoctor,
  loginDoctor as libLogin,
  registerDoctor as libRegister,
  RegisterInput,
  updateProfile as libUpdate,
} from "../lib/auth";

type AuthContextValue = {
  user: DoctorProfile | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<DoctorProfile>;
  register: (input: RegisterInput) => Promise<DoctorProfile>;
  logout: () => void;
  refresh: () => void;
  updateProfile: (
    patch: Partial<Omit<DoctorProfile, "email" | "createdAt">>,
  ) => DoctorProfile;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<DoctorProfile | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setUser(getCurrentUser());
  }, []);

  useEffect(() => {
    refresh();
    setLoading(false);
  }, [refresh]);

  const login = useCallback(async (email: string, password: string) => {
    const u = await libLogin(email, password);
    setUser(u);
    return u;
  }, []);

  const register = useCallback(async (input: RegisterInput) => {
    const u = await libRegister(input);
    setUser(u);
    return u;
  }, []);

  const logout = useCallback(() => {
    logoutDoctor();
    setUser(null);
  }, []);

  const updateProfile = useCallback(
    (patch: Partial<Omit<DoctorProfile, "email" | "createdAt">>) => {
      if (!user) throw new Error("Not signed in.");
      const u = libUpdate(user.email, patch);
      setUser(u);
      return u;
    },
    [user],
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, login, register, logout, refresh, updateProfile }),
    [user, loading, login, register, logout, refresh, updateProfile],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
