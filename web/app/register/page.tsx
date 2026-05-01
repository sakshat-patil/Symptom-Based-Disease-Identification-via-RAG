"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "../components/AuthProvider";

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

export default function RegisterPage() {
  const { register, user, loading } = useAuth();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [fullName, setFullName] = useState("");
  const [specialty, setSpecialty] = useState(SPECIALTIES[1]);
  const [hospital, setHospital] = useState("");
  const [npi, setNpi] = useState("");
  const [yearsExperience, setYearsExperience] = useState<number>(0);
  const [bio, setBio] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && user) router.replace("/app");
  }, [loading, user, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (password !== confirm) {
      setErr("Passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      await register({
        email,
        password,
        fullName,
        specialty,
        hospital,
        npi,
        yearsExperience,
        bio,
      });
      router.replace("/app");
    } catch (e: any) {
      setErr(e.message ?? "Could not create account.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <Link href="/" className="auth__back" aria-label="Back to landing">
        ← Back
      </Link>
      <div className="auth__card auth__card--wide">
        <div className="auth__head">
          <div className="topbar__logo">R</div>
          <div>
            <h1 className="auth__title">Create a doctor account</h1>
            <p className="auth__sub">
              We use your specialty and experience to tailor the
              differential summary tone in the diagnostic tool.
            </p>
          </div>
        </div>
        <form className="auth__form" onSubmit={onSubmit}>
          <div className="auth__grid auth__grid--2">
            <label className="auth__label">
              Full name
              <input
                type="text"
                required
                autoComplete="name"
                className="input"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Alex Kim"
              />
            </label>
            <label className="auth__label">
              Email
              <input
                type="email"
                required
                autoComplete="email"
                className="input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="dr.kim@example.com"
              />
            </label>
          </div>

          <div className="auth__grid auth__grid--2">
            <label className="auth__label">
              Password
              <input
                type="password"
                required
                minLength={6}
                autoComplete="new-password"
                className="input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 6 characters"
              />
            </label>
            <label className="auth__label">
              Confirm password
              <input
                type="password"
                required
                autoComplete="new-password"
                className="input"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="Re-enter password"
              />
            </label>
          </div>

          <div className="auth__grid auth__grid--2">
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

          <div className="auth__grid auth__grid--2">
            <label className="auth__label">
              Hospital / clinic
              <input
                type="text"
                className="input"
                value={hospital}
                onChange={(e) => setHospital(e.target.value)}
                placeholder="SJSU Health Center"
              />
            </label>
            <label className="auth__label">
              NPI <span className="auth__optional">(optional)</span>
              <input
                type="text"
                className="input"
                value={npi}
                onChange={(e) => setNpi(e.target.value)}
                placeholder="10-digit NPI"
              />
            </label>
          </div>

          <label className="auth__label">
            Bio <span className="auth__optional">(optional)</span>
            <textarea
              className="input auth__textarea"
              value={bio}
              onChange={(e) => setBio(e.target.value)}
              placeholder="A short note about your practice or interests."
              rows={3}
            />
          </label>

          {err && <div className="error-banner">{err}</div>}
          <button
            type="submit"
            className="btn btn--primary"
            disabled={busy}
          >
            {busy ? "Creating account…" : "Create account"}
          </button>
        </form>
        <div className="auth__foot">
          Already registered?{" "}
          <Link href="/login" className="auth__link">
            Sign in instead
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
