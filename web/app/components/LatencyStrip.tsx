"use client";

interface Props {
  latency: Record<string, number>;
  vectorStore: string;
  explainer: string;
  alpha: number;
  mode: string;
  backend: string;
}

export function LatencyStrip(p: Props) {
  const total = p.latency["total_ms"] ?? 0;
  const stages = Object.entries(p.latency).filter(([k]) => k !== "total_ms");
  return (
    <div className="latency">
      <span className="latency__total">
        {total.toFixed(0)}
        <small>ms total</small>
      </span>
      <span className="latency__sep" />
      {stages.map(([k, v]) => (
        <span key={k} className="latency__chip">
          <span>{k.replace(/_ms$/, "")}</span>
          {v.toFixed(0)}
          <small style={{ color: "var(--ink-4)" }}>ms</small>
        </span>
      ))}
      <span className="latency__meta">
        <span className="pill pill--mono">{p.backend}</span>
        <span className="pill pill--mono">mode: {p.mode}</span>
        <span className="pill pill--mono">α {p.alpha.toFixed(2)}</span>
        <span className="pill pill--mono">{p.vectorStore}</span>
        <span className="pill pill--mono">{p.explainer}</span>
      </span>
    </div>
  );
}
