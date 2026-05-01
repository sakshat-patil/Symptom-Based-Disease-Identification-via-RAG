"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../components/AuthProvider";
import { AuthGuard } from "../components/AuthGuard";
import { UserMenu } from "../components/UserMenu";
import { Icon } from "../components/Icon";
import {
  DiagnosisHistoryEntry,
  clearHistory,
  readHistory,
} from "../lib/auth";

const SPECIALTIES = [
  "General practice",
  "Internal medicine",
  "Cardiology",
  "Pulmonology",
  "Endocrinology",
  "Gastroenterology",
  "Neurology",
  "Hematology",
  "Infectious disease",
  "Rheumatology",
  "Nephrology",
  "Dermatology",
  "Psychiatry",
  "Emergency medicine",
  "Family medicine",
  "Other",
];

export default function ProfilePage() {
  return (
    <AuthGuard>
      <ProfileContent />
    </AuthGuard>
  );
}

function ProfileContent() {
  const { user, updateProfile } = useAuth();
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [editing, setEditing] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const [fullName, setFullName] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [hospital, setHospital] = useState("");
  const [npi, setNpi] = useState("");
  const [yearsExperience, setYearsExperience] = useState(0);
  const [bio, setBio] = useState("");

  const [history, setHistory] = useState<DiagnosisHistoryEntry[]>([]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!user) return;
    setFullName(user.fullName);
    setSpecialty(user.specialty || SPECIALTIES[1]);
    setHospital(user.hospital);
    setNpi(user.npi);
    setYearsExperience(user.yearsExperience);
    setBio(user.bio);
    setHistory(readHistory(user.email));
  }, [user]);

  const initials = useMemo(() => {
    if (!user) return "DR";
    const parts = user.fullName.split(/\s+/).filter(Boolean).slice(0, 2);
    return parts.map((p) => p[0]?.toUpperCase() ?? "").join("") || "DR";
  }, [user]);

  function onSave(e: React.FormEvent) {
    e.preventDefault();
    updateProfile({
      fullName,
      specialty,
      hospital,
      npi,
      yearsExperience,
      bio,
    });
    setEditing(false);
    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1600);
  }

  function onClearHistory() {
    if (!user) return;
    if (!confirm("Clear all diagnosis history? This cannot be undone.")) return;
    clearHistory(user.email);
    setHistory([]);
  }

  if (!user) return null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <Link href="/" className="topbar__logo-link" aria-label="Home">
            <div className="topbar__logo">R</div>
          </Link>
          <div>
            <div className="topbar__title">
              Record-Based Medical Diagnostic Assistant
            </div>
            <div className="topbar__sub">
              CMPE 255, San Jose State University
            </div>
          </div>
        </div>
        <nav className="topbar__nav">
          <Link href="/app" className="topbar__navlink">
            Diagnose
          </Link>
          <Link href="/insights" className="topbar__navlink">
            Insights
          </Link>
          <Link
            href="/profile"
            className="topbar__navlink topbar__navlink--active"
          >
            Profile
          </Link>
        </nav>
        <div className="topbar__spacer" />
        <button
          className="icon-btn"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Toggle theme"
          aria-label="Toggle theme"
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={14} />
        </button>
        <UserMenu />
      </header>

      <main className="profile">
        <div className="profile__head card">
          <div className="profile__avatar">{initials}</div>
          <div className="profile__head-meta">
            <h1 className="profile__name">{user.fullName}</h1>
            <div className="profile__email">{user.email}</div>
            <div className="profile__chips">
              {user.specialty && (
                <span className="pill pill--accent">{user.specialty}</span>
              )}
              {user.hospital && <span className="pill">{user.hospital}</span>}
              {user.yearsExperience > 0 && (
                <span className="pill pill--mono">
                  {user.yearsExperience} yrs experience
                </span>
              )}
              {user.npi && (
                <span className="pill pill--mono">NPI {user.npi}</span>
              )}
            </div>
          </div>
          <div className="profile__head-actions">
            {editing ? (
              <button
                className="btn"
                onClick={() => setEditing(false)}
                type="button"
              >
                Cancel
              </button>
            ) : (
              <button
                className="btn"
                onClick={() => setEditing(true)}
                type="button"
              >
                Edit profile
              </button>
            )}
            {savedFlash && (
              <span className="profile__saved" role="status">
                Saved
              </span>
            )}
          </div>
        </div>

        <div className="profile__grid">
          <section className="card profile__section">
            <header className="profile__section-head">
              <h2 className="profile__section-title">Profile details</h2>
              <p className="profile__section-sub">
                Editable on this device. Stored locally only.
              </p>
            </header>

            {editing ? (
              <form className="auth__form" onSubmit={onSave}>
                <div className="auth__grid auth__grid--2">
                  <label className="auth__label">
                    Full name
                    <input
                      type="text"
                      className="input"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                    />
                  </label>
                  <label className="auth__label">
                    Specialty
                    <select
                      className="select"
                      value={specialty}
                      onChange={(e) => setSpecialty(e.target.value)}
                    >
                      {SPECIALTIES.map((s) => (
                        <option key={s} value={s}>
                          {s}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="auth__grid auth__grid--2">
                  <label className="auth__label">
                    Hospital / clinic
                    <input
                      type="text"
                      className="input"
                      value={hospital}
                      onChange={(e) => setHospital(e.target.value)}
                    />
                  </label>
                  <label className="auth__label">
                    Years of experience
                    <input
                      type="number"
                      min={0}
                      max={70}
                      className="input"
                      value={yearsExperience}
                      onChange={(e) =>
                        setYearsExperience(Number(e.target.value) || 0)
                      }
                    />
                  </label>
                </div>
                <label className="auth__label">
                  NPI{" "}
                  <span className="auth__optional">(optional)</span>
                  <input
                    type="text"
                    className="input"
                    value={npi}
                    onChange={(e) => setNpi(e.target.value)}
                  />
                </label>
                <label className="auth__label">
                  Bio
                  <textarea
                    className="input auth__textarea"
                    rows={4}
                    value={bio}
                    onChange={(e) => setBio(e.target.value)}
                  />
                </label>
                <button type="submit" className="btn btn--primary">
                  Save changes
                </button>
              </form>
            ) : (
              <dl className="profile__dl">
                <div>
                  <dt>Full name</dt>
                  <dd>{user.fullName}</dd>
                </div>
                <div>
                  <dt>Specialty</dt>
                  <dd>{user.specialty || "—"}</dd>
                </div>
                <div>
                  <dt>Hospital / clinic</dt>
                  <dd>{user.hospital || "—"}</dd>
                </div>
                <div>
                  <dt>Years of experience</dt>
                  <dd>
                    {user.yearsExperience > 0
                      ? `${user.yearsExperience} years`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>NPI</dt>
                  <dd>{user.npi || "—"}</dd>
                </div>
                <div>
                  <dt>Member since</dt>
                  <dd>
                    {new Date(user.createdAt).toLocaleDateString(undefined, {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </dd>
                </div>
                <div className="profile__dl-wide">
                  <dt>Bio</dt>
                  <dd>
                    {user.bio || (
                      <span className="profile__muted">
                        No bio yet. Click <em>Edit profile</em> to add one.
                      </span>
                    )}
                  </dd>
                </div>
              </dl>
            )}
          </section>

          <section className="card profile__section">
            <header className="profile__section-head">
              <h2 className="profile__section-title">Recent diagnoses</h2>
              <p className="profile__section-sub">
                {history.length === 0
                  ? "No queries yet on this device."
                  : `Last ${history.length} ${history.length === 1 ? "query" : "queries"} you ran.`}
              </p>
            </header>
            {history.length === 0 ? (
              <div className="profile__empty">
                <p className="profile__muted">
                  Run a differential and your history will appear here.
                </p>
                <Link
                  href="/app"
                  className="btn btn--primary profile__cta"
                >
                  Open the diagnostic tool
                </Link>
              </div>
            ) : (
              <>
                <ul className="profile__history">
                  {history.map((h, i) => (
                    <li key={i} className="profile__history-row">
                      <div className="profile__history-top">
                        <span className="profile__history-disease">
                          {h.topDisease
                            ? h.topDisease.replace(/_/g, " ")
                            : "No top diagnosis"}
                        </span>
                        {h.fusedScore !== null && (
                          <span className="pill pill--mono">
                            fused {h.fusedScore.toFixed(3)}
                          </span>
                        )}
                      </div>
                      <div className="profile__history-symptoms">
                        {h.symptoms.slice(0, 6).map((s) => (
                          <span key={s} className="chip">
                            {s.replace(/_/g, " ")}
                          </span>
                        ))}
                        {h.symptoms.length > 6 && (
                          <span className="profile__muted">
                            +{h.symptoms.length - 6} more
                          </span>
                        )}
                      </div>
                      <div className="profile__history-meta">
                        {new Date(h.ts).toLocaleString()}
                      </div>
                    </li>
                  ))}
                </ul>
                <div className="profile__history-foot">
                  <button
                    className="btn"
                    onClick={onClearHistory}
                    type="button"
                  >
                    Clear history
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
