"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./Icon";
import { api, ExplainSymptomResponse } from "../api";

interface Props {
  available: string[];
  selected: string[];
  onChange: (next: string[]) => void;
}

const PRESETS: { key: string; label: string; symptoms: string[]; gloss: string }[] = [
  { key: "cardiac",   label: "Cardiac event",   symptoms: ["chest_pain", "lightheadedness", "sweating", "breathlessness"], gloss: "Suspected MI" },
  { key: "flu",       label: "Influenza-like",  symptoms: ["chills", "high_fever", "muscle_pain", "headache"],            gloss: "Viral syndrome" },
  { key: "hep",       label: "Hepatitis-like",  symptoms: ["yellowish_skin", "dark_urine", "fatigue", "abdominal_pain"],   gloss: "Hepatic" },
  { key: "migraine",  label: "Migraine-like",   symptoms: ["headache", "blurred_and_distorted_vision", "irritability", "stiff_neck"], gloss: "Neurologic" },
  { key: "dengue",    label: "Dengue-like",     symptoms: ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"], gloss: "Tropical fever" },
  { key: "pneumonia", label: "Pneumonia-like",  symptoms: ["cough", "high_fever", "breathlessness", "phlegm", "chest_pain"], gloss: "Lower respiratory" },
];

// In-component cache so re-clicking a chip doesn't re-fetch.
const explainCache: Record<string, ExplainSymptomResponse> = {};

export function SymptomPicker({ available, selected, onChange }: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Suggested next symptoms (FP-Growth backed)
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loadingSugs, setLoadingSugs] = useState(false);

  // Preset hover preview
  const [hoverPreset, setHoverPreset] = useState<string | null>(null);

  // Chip to AI explain popover
  const [explainTok, setExplainTok] = useState<string | null>(null);
  const [explainData, setExplainData] = useState<ExplainSymptomResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const matches = useMemo(() => {
    const lower = q.trim().toLowerCase();
    if (!lower) return [];
    return available
      .filter((s) => s.includes(lower) && !selected.includes(s))
      .slice(0, 8);
  }, [q, available, selected]);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // Fetch suggestions whenever the chip set changes (and is non-empty).
  useEffect(() => {
    if (selected.length === 0) {
      setSuggestions([]);
      return;
    }
    let cancelled = false;
    setLoadingSugs(true);
    api
      .suggest(selected)
      .then((r) => {
        if (cancelled) return;
        setSuggestions(
          r.suggestions
            .map((s) => s.symptom)
            .filter((s) => !selected.includes(s))
            .slice(0, 5)
        );
      })
      .catch(() => !cancelled && setSuggestions([]))
      .finally(() => !cancelled && setLoadingSugs(false));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const add = (s: string) => {
    if (!selected.includes(s) && selected.length < 15) onChange([...selected, s]);
    setQ("");
    setOpen(false);
  };
  const remove = (s: string) => onChange(selected.filter((x) => x !== s));

  const activePreset = PRESETS.find(
    (p) =>
      p.symptoms.length === selected.length &&
      p.symptoms.every((s) => selected.includes(s))
  )?.key;

  async function explainChip(tok: string) {
    setExplainTok(tok);
    if (explainCache[tok]) {
      setExplainData(explainCache[tok]);
      return;
    }
    setExplainData(null);
    setExplainLoading(true);
    try {
      const r = await api.explainSymptom(tok);
      explainCache[tok] = r;
      setExplainData(r);
    } catch {
      setExplainData(null);
    } finally {
      setExplainLoading(false);
    }
  }

  return (
    <>
      <div>
        <div className="section-label">
          Patient symptoms
          <span className="count">
            {selected.length}/15, {available.length} tokens
          </span>
        </div>
        <div className="card" style={{ padding: 10 }}>
          <p style={{ margin: 0, marginBottom: 8, fontSize: 12, color: "var(--ink-3)" }}>
            Pick a preset, search, or accept a suggestion. Click any chip for a clinical gloss.
          </p>

          {selected.length > 0 && (
            <div className="chips" style={{ marginBottom: 8 }}>
              {selected.map((c) => (
                <span
                  className="chip chip--clickable"
                  key={c}
                  onClick={() => explainChip(c)}
                  title="Click to explain"
                >
                  {c.replace(/_/g, " ")}
                  <button
                    type="button"
                    className="chip__x"
                    onClick={(e) => {
                      e.stopPropagation();
                      remove(c);
                    }}
                    aria-label={`remove ${c}`}
                  >
                    <Icon name="x" size={10} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <div ref={wrapRef} style={{ position: "relative" }}>
            <div style={{ position: "relative" }}>
              <span style={{ position: "absolute", left: 9, top: 8, color: "var(--ink-4)" }}>
                <Icon name="search" size={14} />
              </span>
              <input
                className="input"
                style={{ paddingLeft: 30 }}
                placeholder="Search 131 symptom tokens…"
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setOpen(true);
                }}
                onFocus={() => setOpen(true)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && matches[0]) {
                    e.preventDefault();
                    add(matches[0]);
                  }
                }}
              />
            </div>
            {open && matches.length > 0 && (
              <div className="autocomplete">
                {matches.map((m) => (
                  <div
                    key={m}
                    className="autocomplete__item"
                    onClick={() => add(m)}
                  >
                    <span>{m.replace(/_/g, " ")}</span>
                    <small>+ add</small>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Suggested next symptoms */}
      {selected.length > 0 && (
        <div>
          <div className="section-label">
            <Icon name="sparkle" size={12} />
            Suggested next
            <span className="count">
              {loadingSugs ? "…" : "from FP-Growth co-occurrence"}
            </span>
          </div>
          {suggestions.length > 0 ? (
            <div className="chips">
              {suggestions.map((s) => (
                <button
                  key={s}
                  className="chip chip--ghost"
                  onClick={() => add(s)}
                  title={`Add "${s.replace(/_/g, " ")}"`}
                >
                  + {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: 12, color: "var(--ink-4)", margin: 0 }}>
              {loadingSugs ? "Computing co-occurrence…" : "No co-occurring symptoms found."}
            </p>
          )}
        </div>
      )}

      <div>
        <div className="section-label">Clinical presets</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {PRESETS.map((p) => {
            const active = activePreset === p.key;
            return (
              <div key={p.key} style={{ position: "relative" }}>
                <button
                  className="btn"
                  style={{
                    width: "100%",
                    justifyContent: "flex-start",
                    fontSize: 12,
                    borderColor: active ? "var(--accent)" : "var(--border)",
                    background: active ? "var(--accent-soft)" : "var(--surface)",
                    color: active ? "var(--accent-strong)" : "var(--ink-2)",
                  }}
                  onClick={() => onChange(p.symptoms)}
                  onMouseEnter={() => setHoverPreset(p.key)}
                  onMouseLeave={() => setHoverPreset(null)}
                >
                  {p.label}
                </button>
                {hoverPreset === p.key && (
                  <div className="preset-preview">
                    <div className="preset-preview__title">
                      {p.label} <span>{p.gloss}</span>
                    </div>
                    {p.symptoms.map((s) => (
                      <div key={s} className="preset-preview__row">
                        {s.replace(/_/g, " ")}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* AI symptom explainer modal, portaled to document.body so it
          escapes the rail's sticky stacking context (otherwise it renders
          behind the main column). */}
      {explainTok && typeof window !== "undefined" && createPortal(
        <div className="modal-backdrop" onClick={() => setExplainTok(null)}>
          <div
            className="modal card"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal__head">
              <span className="pill pill--accent pill--mono">
                {explainTok.replace(/_/g, " ")}
              </span>
              <button
                className="icon-btn"
                style={{ width: 28, height: 28 }}
                onClick={() => setExplainTok(null)}
                aria-label="close"
              >
                <Icon name="x" size={12} />
              </button>
            </div>
            {explainLoading ? (
              <p style={{ fontSize: 13, color: "var(--ink-3)", margin: 0 }}>
                Querying clinical reference…
              </p>
            ) : explainData ? (
              <>
                <p style={{ fontSize: 13, color: "var(--ink), inherit)", margin: "0 0 10px", lineHeight: 1.5 }}>
                  {explainData.explanation}
                </p>
                {explainData.synonyms.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
                      Clinical synonyms
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {explainData.synonyms.map((s) => (
                        <span key={s} className="pill pill--mono">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
                {explainData.top_diseases.length > 0 && (
                  <div>
                    <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
                      Most predictive of
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {explainData.top_diseases.map((d) => (
                        <span key={d} className="pill pill--accent">{d}</span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="modal__foot">
                  <span className="pill pill--mono">{explainData.backend}</span>
                </div>
              </>
            ) : (
              <p style={{ fontSize: 13, color: "var(--ink-4)", margin: 0 }}>
                Could not load explanation.
              </p>
            )}
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
