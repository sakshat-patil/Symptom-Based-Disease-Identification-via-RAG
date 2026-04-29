"use client";
import { useState } from "react";
import { Diagnosis, EvidenceCard as EvCardDTO } from "../api";
import { Icon } from "./Icon";

interface Props {
  rank: number;
  diag: Diagnosis;
  maxScore: number;
}

function highlight(sentence: string, terms: string[]) {
  if (!terms || terms.length === 0) return sentence;
  const sorted = [...terms].sort((a, b) => b.length - a.length);
  const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`(${sorted.map(escape).join("|")})`, "gi");
  const parts = sentence.split(re);
  return parts.map((p, i) =>
    sorted.some((t) => p.toLowerCase() === t.toLowerCase()) ? (
      <mark key={i}>{p}</mark>
    ) : (
      <span key={i}>{p}</span>
    )
  );
}

function tierPill(tier: number) {
  return (
    <span className={`pill pill--tier${tier}`}>
      <span className="dot" />
      tier {tier}
    </span>
  );
}

function ExplanationBlock({
  ex,
  backendTag,
}: {
  ex: Diagnosis["explanation"];
  backendTag: string;
}) {
  const rows: { k: keyof typeof ex; label: string; icon: "link" | "scale" | "check" | "warn"; warn?: boolean }[] = [
    { k: "symptom_disease_link", label: "Symptom to disease link", icon: "link" },
    { k: "statistical_prior", label: "Statistical prior", icon: "scale" },
    { k: "evidence_quality", label: "Evidence quality", icon: "check" },
    { k: "whats_missing", label: "What this system can't tell you", icon: "warn", warn: true },
  ];
  return (
    <div className="expl">
      {rows.map((r) => (
        <div
          key={r.k as string}
          className={"expl__row" + (r.warn ? " expl__row--warn" : "")}
        >
          <span className="expl__icon">
            <Icon name={r.icon} size={16} />
          </span>
          <div>
            <div className="expl__label">{r.label}</div>
            <div className="expl__text">{ex[r.k] as string}</div>
          </div>
        </div>
      ))}
      <div className="expl__foot">
        <span className="pill pill--accent pill--mono">{backendTag}</span>
      </div>
    </div>
  );
}

function EvidenceCardBlock({
  card,
  defaultExpanded,
}: {
  card: EvCardDTO;
  defaultExpanded: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  return (
    <div className="evcard">
      <div className="evcard__head">
        <span className="evcard__source">{card.source_label}</span>
        {tierPill(card.source_tier)}
        <span className="pill">{card.passage_type}</span>
        <span className="pill pill--mono">
          spec {(card.specificity * 100).toFixed(0)}%
        </span>
        {typeof card.ce_score === "number" && (
          <span className="pill pill--mono">ce {card.ce_score.toFixed(2)}</span>
        )}
        <button
          className="evcard__toggle"
          onClick={() => setExpanded((e) => !e)}
        >
          <Icon name={expanded ? "chevron-d" : "chevron-r"} size={12} />
          {expanded ? "Hide passage" : "Show passage"}
        </button>
      </div>
      <div className="evcard__focus">{card.focus}</div>
      {card.claims.slice(0, 2).map((c, i) => (
        <div key={i} className="evcard__claim">
          {highlight(c.sentence, c.matched_terms)}
        </div>
      ))}
      {expanded && (
        <div className="evcard__full">
          {card.full_text}
          <div className="evcard__footer">
            <span>passage_id: {card.passage_id}</span>
            <span>sim: {card.retrieval_score.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export function DiagnosisCard({ rank, diag, maxScore }: Props) {
  const isTop = rank === 1;
  const pct = Math.round((diag.fused_score / (maxScore || 1)) * 100);

  return (
    <div className={"card dx" + (isTop ? " dx--top" : "")}>
      <div className="dx__head">
        <div className="dx__rank">{String(rank).padStart(2, "0")}</div>
        <div style={{ minWidth: 0 }}>
          <h3 className="dx__title">{diag.disease_pretty}</h3>
          <div className="dx__scores">
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div className="score-bar" style={{ width: 200 }}>
                <div className="score-bar__fill" style={{ width: `${pct}%` }} />
              </div>
              <span
                className="mono"
                style={{ fontSize: 13, fontWeight: 600, color: "var(--accent-strong)" }}
              >
                {diag.fused_score.toFixed(3)}
              </span>
            </div>
            <span className="dx__score-num">
              <em>Mining</em>
              <strong>{diag.mining_score.toFixed(3)}</strong>
            </span>
            <span className="dx__score-num">
              <em>Retrieval</em>
              <strong>{diag.retrieval_score.toFixed(3)}</strong>
            </span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {isTop && <span className="pill pill--accent">top match</span>}
        </div>
      </div>

      <ExplanationBlock ex={diag.explanation} backendTag={diag.explanation.backend} />

      {diag.matching_rules.length > 0 && (
        <div className="rules">
          <div className="section-label">
            Matching FP-Growth rules
            <span className="count">{diag.matching_rules.length} fired</span>
          </div>
          {diag.matching_rules.slice(0, 3).map((r, i) => (
            <div key={i} className="rule">
              <span className="rule__antecedent">
                {`{${r.antecedent.map((a) => a.replace(/_/g, " ")).join(", ")}}`}
              </span>
              <span className="rule__arrow">to</span>
              <span className="rule__disease">{diag.disease_pretty}</span>
              <span className="rule__stats">
                conf {r.confidence.toFixed(2)}, lift {r.lift.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="ev">
        <div className="section-label">
          Supporting evidence
          <span className="count">
            {diag.evidence_cards.length} card{diag.evidence_cards.length === 1 ? "" : "s"}
          </span>
        </div>
        {diag.evidence_cards.length === 0 ? (
          <p style={{ fontSize: 12, color: "var(--ink-4)", fontStyle: "italic", margin: 0 }}>
            No high-specificity literature evidence found for this candidate. See "Related context" below for nearby passages.
          </p>
        ) : (
          <div className="ev__grid">
            {diag.evidence_cards.map((c, i) => (
              <EvidenceCardBlock
                key={c.passage_id}
                card={c}
                defaultExpanded={isTop && i === 0}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
