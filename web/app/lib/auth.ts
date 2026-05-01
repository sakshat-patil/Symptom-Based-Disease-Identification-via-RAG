/**
 * Mock authentication backed by localStorage.
 *
 * Not a security boundary. The class project demonstrates the UX of
 * doctor sign-up/sign-in/profile but does not claim password security:
 * everything lives in the browser. Real auth would move this to the
 * FastAPI service with httpOnly cookies.
 */

export type DoctorProfile = {
  email: string;
  fullName: string;
  specialty: string;
  hospital: string;
  npi: string;
  yearsExperience: number;
  bio: string;
  createdAt: string;
};

type StoredUser = DoctorProfile & {
  passwordHash: string;
};

export type DiagnosisHistoryEntry = {
  ts: string;
  symptoms: string[];
  topDisease: string | null;
  fusedScore: number | null;
};

const USERS_KEY = "med_rag_users_v1";
const SESSION_KEY = "med_rag_session_v1";
const HISTORY_PREFIX = "med_rag_history_v1::";

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

async function hashPassword(password: string): Promise<string> {
  if (!isBrowser() || !window.crypto?.subtle) {
    // Server-side or environment without subtle.crypto. The auth flow
    // never runs server-side in this app, but TypeScript needs a path.
    return password;
  }
  const enc = new TextEncoder().encode(password);
  const buf = await window.crypto.subtle.digest("SHA-256", enc);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function readUsers(): Record<string, StoredUser> {
  if (!isBrowser()) return {};
  try {
    const raw = localStorage.getItem(USERS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, StoredUser>) : {};
  } catch {
    return {};
  }
}

function writeUsers(users: Record<string, StoredUser>): void {
  if (!isBrowser()) return;
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export type RegisterInput = {
  email: string;
  password: string;
  fullName: string;
  specialty: string;
  hospital: string;
  npi?: string;
  yearsExperience?: number;
  bio?: string;
};

export async function registerDoctor(
  input: RegisterInput,
): Promise<DoctorProfile> {
  const email = input.email.trim().toLowerCase();
  if (!email || !input.password || !input.fullName.trim()) {
    throw new Error("Email, password, and full name are required.");
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw new Error("Enter a valid email address.");
  }
  if (input.password.length < 6) {
    throw new Error("Password must be at least 6 characters.");
  }
  const users = readUsers();
  if (users[email]) {
    throw new Error("An account with this email already exists.");
  }
  const passwordHash = await hashPassword(input.password);
  const user: StoredUser = {
    email,
    fullName: input.fullName.trim(),
    specialty: input.specialty.trim(),
    hospital: input.hospital.trim(),
    npi: (input.npi ?? "").trim(),
    yearsExperience: Math.max(0, Math.min(70, input.yearsExperience ?? 0)),
    bio: (input.bio ?? "").trim(),
    createdAt: new Date().toISOString(),
    passwordHash,
  };
  users[email] = user;
  writeUsers(users);
  setSession(email);
  return stripPassword(user);
}

export async function loginDoctor(
  email: string,
  password: string,
): Promise<DoctorProfile> {
  const e = email.trim().toLowerCase();
  if (!e || !password) {
    throw new Error("Email and password are required.");
  }
  const users = readUsers();
  const user = users[e];
  if (!user) {
    throw new Error("No account found for that email.");
  }
  const hashed = await hashPassword(password);
  if (hashed !== user.passwordHash) {
    throw new Error("Incorrect password.");
  }
  setSession(e);
  return stripPassword(user);
}

export function logoutDoctor(): void {
  if (!isBrowser()) return;
  localStorage.removeItem(SESSION_KEY);
}

export function getCurrentEmail(): string | null {
  if (!isBrowser()) return null;
  return localStorage.getItem(SESSION_KEY);
}

export function getCurrentUser(): DoctorProfile | null {
  const email = getCurrentEmail();
  if (!email) return null;
  const users = readUsers();
  const user = users[email];
  if (!user) return null;
  return stripPassword(user);
}

export function updateProfile(
  email: string,
  patch: Partial<Omit<DoctorProfile, "email" | "createdAt">>,
): DoctorProfile {
  const users = readUsers();
  const user = users[email.toLowerCase()];
  if (!user) throw new Error("Account not found.");
  const next: StoredUser = {
    ...user,
    fullName: patch.fullName?.trim() ?? user.fullName,
    specialty: patch.specialty?.trim() ?? user.specialty,
    hospital: patch.hospital?.trim() ?? user.hospital,
    npi: patch.npi?.trim() ?? user.npi,
    yearsExperience:
      patch.yearsExperience !== undefined
        ? Math.max(0, Math.min(70, patch.yearsExperience))
        : user.yearsExperience,
    bio: patch.bio?.trim() ?? user.bio,
  };
  users[email.toLowerCase()] = next;
  writeUsers(users);
  return stripPassword(next);
}

function setSession(email: string): void {
  if (!isBrowser()) return;
  localStorage.setItem(SESSION_KEY, email);
}

function stripPassword(u: StoredUser): DoctorProfile {
  const { passwordHash: _ph, ...rest } = u;
  return rest;
}

// ---------- Per-user diagnosis history (last 25 queries) ----------

export function recordHistory(
  email: string,
  entry: DiagnosisHistoryEntry,
): void {
  if (!isBrowser()) return;
  const key = HISTORY_PREFIX + email.toLowerCase();
  const existing = readHistory(email);
  const next = [entry, ...existing].slice(0, 25);
  localStorage.setItem(key, JSON.stringify(next));
}

export function readHistory(email: string): DiagnosisHistoryEntry[] {
  if (!isBrowser()) return [];
  const key = HISTORY_PREFIX + email.toLowerCase();
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as DiagnosisHistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function clearHistory(email: string): void {
  if (!isBrowser()) return;
  localStorage.removeItem(HISTORY_PREFIX + email.toLowerCase());
}
