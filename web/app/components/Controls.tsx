"use client";
import { useState, ReactNode } from "react";
import { ConfigDTO, SourceDTO } from "../api";
import { Icon } from "./Icon";

interface Props {
  config: ConfigDTO | null;
  sources: SourceDTO[];
  backend: string;
  mode: string;
  alpha: number;
  expandSyn: boolean;
  crossEncoder: boolean;
  explainer: string;
  sourceFilter: string | null;
  passageTypeFilter: string | null;
  topKRetrieval: number;
  relatedTopK: number;
  maxEvidenceCards: number;
  compareMode: boolean;
  onChange: (
    patch: Partial<{
      backend: string;
      mode: string;
      alpha: number;
      expandSyn: boolean;
      crossEncoder: boolean;
      explainer: string;
      sourceFilter: string | null;
      passageTypeFilter: string | null;
      topKRetrieval: number;
      relatedTopK: number;
      maxEvidenceCards: number;
      compareMode: boolean;
    }>
  ) => void;
}

function Collapsible({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon: "sliders" | "filter";
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card collapsible" data-open={open}>
      <div className="collapsible__header" onClick={() => setOpen((o) => !o)}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "var(--ink-3)" }}>
            <Icon name={icon} size={14} />
          </span>
          <span className="collapsible__title">{title}</span>
        </div>
        <span className="collapsible__chev">
          <Icon name="chevron" size={14} />
        </span>
      </div>
      <div className="collapsible__body">{children}</div>
    </div>
  );
}

export function Controls(p: Props) {
  if (!p.config) {
    return (
      <div className="card" style={{ padding: 14, fontSize: 13, color: "var(--ink-4)" }}>
        Loading service config…
      </div>
    );
  }

  const totalPassages = p.sources.reduce((a, s) => a + s.count, 0);

  // Backend availability is constrained by both the vector store dimension
  // and the current mode. In Pinecone mode the index is 3072d so only
  // azure-openai is compatible; mining-only ignores the backend entirely.
  const pineconeMode = p.config.vector_store === "pinecone";
  const backendDisabled = p.mode === "mining-only";
  const visibleBackends = backendDisabled
    ? p.config.backends
    : pineconeMode
    ? p.config.backends.filter((b) => b === "azure-openai")
    : p.config.backends;

  return (
    <>
      <Collapsible title="Pipeline" icon="sliders">
        <div style={{ display: "grid", gap: 10 }}>
          <label
            className="checkbox"
            title="Run azure-openai (3072d Pinecone) and pubmedbert (768d Pinecone) in parallel and show results side by side."
          >
            <input
              type="checkbox"
              checked={p.compareMode}
              onChange={(e) => p.onChange({ compareMode: e.target.checked })}
            />
            Compare backends
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--ink-4)" }}>
              azure ↔ pubmedbert
            </span>
          </label>
          <div>
            <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
              Backend
              {p.compareMode && (
                <span className="count">overridden by compare mode</span>
              )}
              {!p.compareMode && backendDisabled && (
                <span className="count">unused in mining-only</span>
              )}
            </div>
            <select
              className="select"
              value={p.backend}
              disabled={backendDisabled || p.compareMode}
              onChange={(e) => p.onChange({ backend: e.target.value })}
            >
              {visibleBackends.map((b) => (
                <option key={b} value={b}>
                  {b}
                  {b === "azure-openai" && pineconeMode
                    ? ", pinecone (3072d)"
                    : ""}
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
              Mode
            </div>
            <select
              className="select"
              value={p.mode}
              onChange={(e) => p.onChange({ mode: e.target.value })}
            >
              <option value="fused">fused</option>
              <option value="mining-only">mining-only</option>
              <option value="retrieval-only">retrieval-only</option>
            </select>
          </div>

          <div>
            <div className="kv" style={{ marginBottom: 4 }}>
              <span className="section-label" style={{ margin: 0 }}>
                Fusion weight (α)
              </span>
              <span className="kv__v">{p.alpha.toFixed(2)}</span>
            </div>
            <div
              className="slider-row"
              title="0 = mining-only, 1 = retrieval-only, default 0.30 (alpha-sweep optimum)"
            >
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={p.alpha}
                disabled={p.mode !== "fused"}
                className="slider"
                onChange={(e) => p.onChange({ alpha: parseFloat(e.target.value) })}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--ink-4)", marginTop: 4 }}>
              <span>← Mining</span>
              <span>Retrieval to</span>
            </div>
          </div>

          <label
            className="checkbox"
            title="Map symptom tokens to clinical synonyms before encoding (muscle_pain to myalgia, …)"
          >
            <input
              type="checkbox"
              checked={p.expandSyn}
              onChange={(e) => p.onChange({ expandSyn: e.target.checked })}
            />
            Synonym expansion
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--ink-4)" }}>
              UMLS-style
            </span>
          </label>
          <label
            className="checkbox"
            title="Re-score top-K (query, passage) pairs jointly with a cross-encoder. Slower but sharper at K≤5."
          >
            <input
              type="checkbox"
              checked={p.crossEncoder}
              onChange={(e) => p.onChange({ crossEncoder: e.target.checked })}
            />
            Cross-encoder rerank
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--ink-4)" }}>
              {p.config.vector_store === "pinecone" ? "Pinecone" : "local"}
            </span>
          </label>

          <div>
            <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
              Explainer
            </div>
            <select
              className="select"
              value={p.explainer}
              onChange={(e) => p.onChange({ explainer: e.target.value })}
            >
              {p.config.explainers.map((b) => (
                <option key={b} value={b}>
                  {b === "openai" ? "openai:gpt-4o-mini" : b}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Collapsible>

      <Collapsible title="Tuning knobs" icon="sliders" defaultOpen={false}>
        <div style={{ display: "grid", gap: 12 }}>
          <NumericKnob
            label="Vector-search top-K"
            hint="How deep to search the vector store before disease attribution. Higher = more recall, slower."
            value={p.topKRetrieval}
            min={5}
            max={80}
            step={5}
            onChange={(v) => p.onChange({ topKRetrieval: v })}
          />
          <NumericKnob
            label="Related-context size"
            hint="How many 'nearby passages' to surface in the related-context rail at the bottom of the page."
            value={p.relatedTopK}
            min={1}
            max={15}
            step={1}
            onChange={(v) => p.onChange({ relatedTopK: v })}
          />
          <NumericKnob
            label="Evidence cards per diagnosis"
            hint="Cap on how many evidence cards each ranked diagnosis renders. Lower = tighter UI; higher = more sources."
            value={p.maxEvidenceCards}
            min={1}
            max={10}
            step={1}
            onChange={(v) => p.onChange({ maxEvidenceCards: v })}
          />
        </div>
      </Collapsible>

      <Collapsible title="Evidence filters" icon="filter" defaultOpen={false}>
        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
              Source of truth
            </div>
            <select
              className="select"
              value={p.sourceFilter ?? ""}
              onChange={(e) =>
                p.onChange({ sourceFilter: e.target.value || null })
              }
            >
              <option value="">
                All sources ({totalPassages.toLocaleString()})
              </option>
              {p.sources.map((s) => (
                <option key={s.source_id} value={s.source_id}>
                  {s.label}, {s.count.toLocaleString()} passages
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="section-label" style={{ margin: 0, marginBottom: 4 }}>
              Passage type
            </div>
            <select
              className="select"
              value={p.passageTypeFilter ?? ""}
              onChange={(e) =>
                p.onChange({ passageTypeFilter: e.target.value || null })
              }
            >
              <option value="">All types</option>
              {[
                "symptoms",
                "diagnosis",
                "treatment",
                "causes",
                "complications",
                "prevention",
                "genetics",
                "overview",
                "general",
              ].map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Collapsible>
    </>
  );
}

// Slider + numeric readout for an integer knob. Tooltip shows the full
// hint on hover so the rail stays compact.
function NumericKnob({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="kv" style={{ marginBottom: 4 }}>
        <span
          className="section-label"
          style={{ margin: 0 }}
          title={hint}
        >
          {label}
        </span>
        <span className="kv__v">{value}</span>
      </div>
      <div className="slider-row" title={hint}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          className="slider"
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
        />
      </div>
      <p
        style={{
          fontSize: 10.5,
          color: "var(--ink-4)",
          margin: "4px 0 0",
          lineHeight: 1.35,
        }}
      >
        {hint}
      </p>
    </div>
  );
}
