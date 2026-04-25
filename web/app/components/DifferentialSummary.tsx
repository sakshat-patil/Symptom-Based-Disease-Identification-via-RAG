"use client";
import { useEffect, useState } from "react";
import { api, DifferentialResponse } from "../api";
import { Icon } from "./Icon";

interface Props {
  symptoms: string[];
  diagnoses: {
    disease: string;
    fused_score: number;
    mining_score: number;
    retrieval_score: number;
  }[];
  alpha: number;
  mode: string;
}

export function DifferentialSummary(p: Props) {
  const [data, setData] = useState<DifferentialResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Re-fetch only when the (symptoms, top-3) tuple changes.
  const top3 = p.diagnoses.slice(0, 3);
  const cacheKey = JSON.stringify({
    s: [...p.symptoms].sort(),
    d: top3.map((d) => d.disease).sort(),
    a: p.alpha.toFixed(2),
    m: p.mode,
  });

  useEffect(() => {
    if (top3.length === 0) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    api
      .differential({
        symptoms: p.symptoms,
        candidates: top3,
        alpha: p.alpha,
        mode: p.mode,
      })
      .then((r) => !cancelled && setData(r))
      .catch((e) => !cancelled && setErr(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheKey]);

  if (top3.length === 0) return null;

  return (
    <div className="card diff">
      <div className="diff__head">
        <Icon name="sparkle" size={14} />
        <span className="diff__title">Differential summary</span>
      </div>
      {loading && (
        <div className="diff--loading">
          <span className="spinner" style={{ borderTopColor: "var(--accent-strong)", borderColor: "oklch(from var(--accent) l c h / 0.25)" }} />
          Synthesising clinician-style summary…
        </div>
      )}
      {!loading && data && (
        <>
          <p className="diff__body">{data.summary}</p>
          <div className="diff__foot">
            <span className="pill pill--mono">{data.backend}</span>
          </div>
        </>
      )}
      {!loading && err && (
        <p style={{ color: "var(--ink-4)", fontSize: 12, margin: 0 }}>
          Could not generate summary: {err}
        </p>
      )}
    </div>
  );
}
