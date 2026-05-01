"use client";
import { useEffect, useState } from "react";
import {
  api,
  ConfigDTO,
  DiagnoseRequest,
  DiagnoseResponse,
  InsightsResponse,
  PipelineStage,
  SourceDTO,
} from "../api";
import { SymptomPicker } from "../components/SymptomPicker";
import { Controls } from "../components/Controls";
import { DiagnosisCard } from "../components/DiagnosisCard";
import { RelatedContextPanel } from "../components/RelatedContext";
import { LatencyStrip } from "../components/LatencyStrip";
import { PipelineTimeline } from "../components/PipelineTimeline";
import { DifferentialSummary } from "../components/DifferentialSummary";
import Link from "next/link";
import { Icon } from "../components/Icon";
import { AuthGuard } from "../components/AuthGuard";
import { UserMenu } from "../components/UserMenu";
import { useAuth } from "../components/AuthProvider";
import { recordHistory } from "../lib/auth";

export default function AppPage() {
  return (
    <AuthGuard>
      <DiagnosticTool />
    </AuthGuard>
  );
}

function DiagnosticTool() {
  const { user } = useAuth();
  const [config, setConfig] = useState<ConfigDTO | null>(null);
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [allSymptoms, setAllSymptoms] = useState<string[]>([]);
  const [sources, setSources] = useState<SourceDTO[]>([]);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);

  const [backend, setBackend] = useState("pubmedbert");
  const [mode, setMode] = useState("fused");
  const [alpha, setAlpha] = useState(0.3);
  const [expandSyn, setExpandSyn] = useState(true);
  const [crossEncoder, setCrossEncoder] = useState(false);
  const [explainer, setExplainer] = useState("template");
  const [sourceFilter, setSourceFilter] = useState<string | null>(null);
  const [passageTypeFilter, setPassageTypeFilter] = useState<string | null>(null);
  const [topKRetrieval, setTopKRetrieval] = useState(30);
  const [relatedTopK, setRelatedTopK] = useState(5);
  const [maxEvidenceCards, setMaxEvidenceCards] = useState(5);

  const [resp, setResp] = useState<DiagnoseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamStages, setStreamStages] = useState<PipelineStage[]>([]);

  const [compareMode, setCompareMode] = useState(false);
  const [respB, setRespB] = useState<DiagnoseResponse | null>(null);
  const [loadingB, setLoadingB] = useState(false);
  const [streamStagesB, setStreamStagesB] = useState<PipelineStage[]>([]);
  const [errorB, setErrorB] = useState<string | null>(null);
  const [heroExiting, setHeroExiting] = useState(false);

  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!config) return;
    if (
      config.vector_store === "pinecone" &&
      mode !== "mining-only" &&
      backend !== "azure-openai" &&
      config.backends.includes("azure-openai")
    ) {
      setBackend("azure-openai");
    }
  }, [config, mode, backend]);

  useEffect(() => {
    (async () => {
      try {
        const [cfg, syms, src, ins] = await Promise.all([
          api.config(),
          api.symptoms(),
          api.sources(),
          api.insights().catch(() => null),
        ]);
        setConfig(cfg);
        setAllSymptoms(syms.symptoms);
        setSources(src.sources);
        setInsights(ins);
        setExplainer(cfg.explainers[0] ?? "template");
        setAlpha(cfg.default_alpha ?? 0.3);
        if (
          cfg.vector_store === "pinecone" &&
          cfg.backends.includes("azure-openai")
        ) {
          setBackend("azure-openai");
        }
      } catch (e: any) {
        setError(`Could not reach API: ${e.message}`);
      }
    })();
  }, []);

  // Record history once a diagnosis lands.
  useEffect(() => {
    if (!resp || !user) return;
    const top = resp.diagnoses[0];
    recordHistory(user.email, {
      ts: new Date().toISOString(),
      symptoms: resp.query_symptoms,
      topDisease: top?.disease ?? null,
      fusedScore: top?.fused_score ?? null,
    });
  }, [resp, user]);

  async function runDiagnose() {
    if (!symptoms.length) return;
    setError(null);
    setErrorB(null);
    if (!resp && !respB) {
      setHeroExiting(true);
      await new Promise((r) => setTimeout(r, 220));
      setHeroExiting(false);
    }
    setLoading(true);
    setResp(null);
    setStreamStages([]);

    const baseReq: Omit<DiagnoseRequest, "backend"> = {
      symptoms,
      mode,
      alpha,
      expand_synonyms: expandSyn,
      cross_encoder: crossEncoder,
      explainer,
      source_filter: sourceFilter,
      passage_type_filter: passageTypeFilter,
      top_n: 5,
      top_k_retrieval: topKRetrieval,
      related_top_k: relatedTopK,
      max_evidence_cards: maxEvidenceCards,
      trace: true,
    };

    const laneABackend = compareMode ? "azure-openai" : backend;
    const reqA: DiagnoseRequest = { ...baseReq, backend: laneABackend };

    const streamA = api.diagnoseStream(reqA, {
      onStage: (s) => setStreamStages((cur) => [...cur, s]),
      onComplete: (full) => {
        setResp(full);
        setLoading(false);
      },
      onError: (msg) => {
        setError(msg);
        setLoading(false);
      },
    });

    let streamB: Promise<void> | null = null;
    if (compareMode) {
      setLoadingB(true);
      setRespB(null);
      setStreamStagesB([]);
      const reqB: DiagnoseRequest = { ...baseReq, backend: "pubmedbert" };
      streamB = api.diagnoseStream(reqB, {
        onStage: (s) => setStreamStagesB((cur) => [...cur, s]),
        onComplete: (full) => {
          setRespB(full);
          setLoadingB(false);
        },
        onError: (msg) => {
          setErrorB(msg);
          setLoadingB(false);
        },
      });
    } else {
      setRespB(null);
      setStreamStagesB([]);
      setLoadingB(false);
    }

    try {
      await Promise.all([streamA, streamB].filter(Boolean) as Promise<void>[]);
    } catch (e: any) {
      setError((cur) => cur ?? e.message);
      setLoading(false);
      setLoadingB(false);
    }
  }

  const maxScore = resp
    ? Math.max(...resp.diagnoses.map((d) => d.fused_score), 0.001)
    : 1;

  const statusBackend = resp?.used_backend ?? backend;
  const statusVS = config?.vector_store ?? ", ";
  const statusExp =
    resp?.explainer_backend?.replace("openai:", "") ?? explainer;

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
          <Link
            href="/app"
            className="topbar__navlink topbar__navlink--active"
          >
            Diagnose
          </Link>
          <Link href="/insights" className="topbar__navlink">
            Insights
          </Link>
          <Link href="/profile" className="topbar__navlink">
            Profile
          </Link>
        </nav>
        <div className="topbar__spacer" />
        <span className="status">
          <span className="dot" />
          <span className="mono">{statusVS}</span>
          <span style={{ color: "var(--ink-4)" }}>, </span>
          <span className="mono">{statusBackend}</span>
          <span style={{ color: "var(--ink-4)" }}>, </span>
          <span className="mono">{statusExp}</span>
        </span>
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

      <div className="shell">
        <aside className="rail">
          <SymptomPicker
            available={allSymptoms}
            selected={symptoms}
            onChange={setSymptoms}
          />
          <Controls
            config={config}
            sources={sources}
            backend={backend}
            mode={mode}
            alpha={alpha}
            expandSyn={expandSyn}
            crossEncoder={crossEncoder}
            explainer={explainer}
            sourceFilter={sourceFilter}
            passageTypeFilter={passageTypeFilter}
            topKRetrieval={topKRetrieval}
            relatedTopK={relatedTopK}
            maxEvidenceCards={maxEvidenceCards}
            compareMode={compareMode}
            onChange={(p) => {
              if (p.backend !== undefined) setBackend(p.backend);
              if (p.mode !== undefined) setMode(p.mode);
              if (p.alpha !== undefined) setAlpha(p.alpha);
              if (p.expandSyn !== undefined) setExpandSyn(p.expandSyn);
              if (p.crossEncoder !== undefined) setCrossEncoder(p.crossEncoder);
              if (p.explainer !== undefined) setExplainer(p.explainer);
              if (p.sourceFilter !== undefined) setSourceFilter(p.sourceFilter);
              if (p.passageTypeFilter !== undefined)
                setPassageTypeFilter(p.passageTypeFilter);
              if (p.topKRetrieval !== undefined)
                setTopKRetrieval(p.topKRetrieval);
              if (p.relatedTopK !== undefined) setRelatedTopK(p.relatedTopK);
              if (p.maxEvidenceCards !== undefined)
                setMaxEvidenceCards(p.maxEvidenceCards);
              if (p.compareMode !== undefined) setCompareMode(p.compareMode);
            }}
          />
          <div style={{ flex: 1 }} />
          <button
            className="btn btn--primary"
            onClick={runDiagnose}
            disabled={loading || !symptoms.length}
          >
            {loading
              ? "Diagnosing…"
              : symptoms.length === 0
              ? "Add a symptom to diagnose"
              : `Diagnose, ${symptoms.length} symptom${symptoms.length === 1 ? "" : "s"}`}
          </button>
          {error && <div className="error-banner">{error}</div>}
        </aside>

        <main className="main">
          {(heroExiting || (!loading && !resp && !error)) && (
            <EmptyState
              hasSymptoms={symptoms.length > 0}
              symptomCount={symptoms.length}
              onTry={runDiagnose}
              exiting={heroExiting}
            />
          )}

          {!compareMode && (loading || resp) && (
            <>
              <PipelineTimeline
                state={loading ? "running" : "done"}
                symptomCount={symptoms.length}
                backend={resp?.used_backend ?? backend}
                vectorStore={
                  resp?.vector_store ?? config?.vector_store ?? "faiss"
                }
                explainer={resp?.explainer_backend ?? explainer}
                alpha={resp?.used_alpha ?? alpha}
                mode={resp?.used_mode ?? mode}
                trace={
                  streamStages.length > 0
                    ? streamStages
                    : resp?.pipeline_trace ?? []
                }
                insights={insights}
                streaming={loading}
              />
              {loading && <SkeletonResults />}
              {!loading && resp && (
                <>
                  <LatencyStrip
                    latency={resp.latency_ms}
                    vectorStore={resp.vector_store}
                    explainer={resp.explainer_backend}
                    alpha={resp.used_alpha}
                    mode={resp.used_mode}
                    backend={resp.used_backend}
                  />
                  <DifferentialSummary
                    symptoms={resp.query_symptoms}
                    diagnoses={resp.diagnoses.map((d) => ({
                      disease: d.disease,
                      fused_score: d.fused_score,
                      mining_score: d.mining_score,
                      retrieval_score: d.retrieval_score,
                    }))}
                    alpha={resp.used_alpha}
                    mode={resp.used_mode}
                  />
                  <h3 className="h-section" style={{ marginTop: 4 }}>
                    Ranked diagnoses
                    <span>
                      top {resp.diagnoses.length}, α{" "}
                      {resp.used_alpha.toFixed(2)}, mode {resp.used_mode}
                    </span>
                  </h3>
                  {resp.diagnoses.map((d, i) => (
                    <DiagnosisCard
                      key={d.disease}
                      rank={i + 1}
                      diag={d}
                      maxScore={maxScore}
                    />
                  ))}
                  <RelatedContextPanel items={resp.related_context} />
                </>
              )}
            </>
          )}

          {compareMode && (loading || loadingB || resp || respB) && (
            <>
              <h3 className="h-section" style={{ marginTop: 0 }}>
                Side-by-side compare
                <span>
                  azure-openai, 3072d  vs.  pubmedbert, 768d
                </span>
              </h3>
              <div className="compare-grid">
                <CompareLane
                  title="azure-openai, 3072d Pinecone"
                  loading={loading}
                  resp={resp}
                  streamStages={streamStages}
                  symptoms={symptoms}
                  alpha={alpha}
                  mode={mode}
                  fallbackBackend="azure-openai"
                  fallbackVectorStore={config?.vector_store ?? "pinecone"}
                  fallbackExplainer={explainer}
                  insights={insights}
                  error={error}
                />
                <CompareLane
                  title="pubmedbert, 768d Pinecone"
                  loading={loadingB}
                  resp={respB}
                  streamStages={streamStagesB}
                  symptoms={symptoms}
                  alpha={alpha}
                  mode={mode}
                  fallbackBackend="pubmedbert"
                  fallbackVectorStore={config?.vector_store ?? "pinecone"}
                  fallbackExplainer={explainer}
                  insights={insights}
                  error={errorB}
                />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function CompareLane(props: {
  title: string;
  loading: boolean;
  resp: DiagnoseResponse | null;
  streamStages: PipelineStage[];
  symptoms: string[];
  alpha: number;
  mode: string;
  fallbackBackend: string;
  fallbackVectorStore: string;
  fallbackExplainer: string;
  insights: InsightsResponse | null;
  error: string | null;
}) {
  const {
    title,
    loading,
    resp,
    streamStages,
    symptoms,
    alpha,
    mode,
    fallbackBackend,
    fallbackVectorStore,
    fallbackExplainer,
    insights,
    error,
  } = props;
  const maxScore = resp
    ? Math.max(...resp.diagnoses.map((d) => d.fused_score), 0.001)
    : 1;
  return (
    <div className="compare-lane">
      <div className="compare-lane__head">
        <h4 className="compare-lane__title">{title}</h4>
        {resp && (
          <span className="pill pill--mono">
            top: {resp.diagnoses[0]?.disease.replace(/_/g, " ")} , {" "}
            {resp.diagnoses[0]?.fused_score.toFixed(3)}
          </span>
        )}
      </div>
      {error && <div className="error-banner">{error}</div>}
      {(loading || resp) && (
        <PipelineTimeline
          state={loading ? "running" : "done"}
          symptomCount={symptoms.length}
          backend={resp?.used_backend ?? fallbackBackend}
          vectorStore={resp?.vector_store ?? fallbackVectorStore}
          explainer={resp?.explainer_backend ?? fallbackExplainer}
          alpha={resp?.used_alpha ?? alpha}
          mode={resp?.used_mode ?? mode}
          trace={
            streamStages.length > 0
              ? streamStages
              : resp?.pipeline_trace ?? []
          }
          insights={insights}
          streaming={loading}
        />
      )}
      {!loading && resp && (
        <>
          <h5 className="compare-lane__section">
            Ranked diagnoses
            <span>top {resp.diagnoses.length}</span>
          </h5>
          {resp.diagnoses.slice(0, 3).map((d, i) => (
            <DiagnosisCard
              key={d.disease}
              rank={i + 1}
              diag={d}
              maxScore={maxScore}
            />
          ))}
        </>
      )}
    </div>
  );
}

function SkeletonResults() {
  return (
    <>
      <div className="skel" style={{ height: 36, marginBottom: 16 }} />
      {[1, 2, 3].map((i) => (
        <div key={i} className="card skel-card">
          <div style={{ display: "flex", gap: 14 }}>
            <div className="skel" style={{ width: 36, height: 24 }} />
            <div style={{ flex: 1 }}>
              <div className="skel" style={{ width: "55%", height: 18, marginBottom: 8 }} />
              <div className="skel" style={{ width: "75%", height: 8 }} />
            </div>
          </div>
          <div className="skel" style={{ height: 110, marginTop: 14 }} />
          <div className="skel" style={{ height: 60, marginTop: 10 }} />
        </div>
      ))}
    </>
  );
}

function EmptyState({
  hasSymptoms,
  symptomCount,
  onTry,
  exiting,
}: {
  hasSymptoms: boolean;
  symptomCount: number;
  onTry: () => void;
  exiting?: boolean;
}) {
  return (
    <div className="empty" data-exit={exiting ? "true" : "false"}>
      <div className="empty__crumbs">
        <span className="dot" />
        <span>{symptomCount} / 15 symptoms</span>
        <span className="empty__crumb-sep">, </span>
        <span>131 vector tokens</span>
        <span className="empty__crumb-sep">, </span>
        <span>23,839 mined rules</span>
      </div>
      <h1 className="empty__title">
        What is the patient <em>presenting&nbsp;with?</em>
      </h1>
      <p className="empty__lede">
        A symptom-first differential. Add what you observe in the rail ,
        the system ranks likely diagnoses against{" "}
        <strong>24,063 mined biomedical passages</strong> and{" "}
        <strong>23,839 association rules</strong>, then explains its
        reasoning with claim-level evidence.
      </p>
      <button
        className="empty__cta"
        onClick={onTry}
        disabled={!hasSymptoms}
      >
        {hasSymptoms
          ? `Run differential, ${symptomCount} symptom${symptomCount === 1 ? "" : "s"}`
          : "Add at least one symptom to begin"}
        <span aria-hidden>to</span>
      </button>
      {!hasSymptoms && (
        <span className="empty__hint">
          Try the <em style={{ fontFamily: "var(--serif)", color: "var(--ink-2)" }}>Cardiac event</em> preset on the left to see a worked example.
        </span>
      )}
    </div>
  );
}
