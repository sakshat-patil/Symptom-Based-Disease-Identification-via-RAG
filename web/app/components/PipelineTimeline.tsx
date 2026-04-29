"use client";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Icon } from "./Icon";
import { InsightsResponse, PipelineStage } from "../api";
import {
  useCountUp,
  useStaggeredReveal,
  useTypewriter,
} from "./animationHooks";

const TIER_COLORS = ["#2f7a4f", "#b88500", "#94a3b8"];

interface Props {
  state: "idle" | "running" | "done";
  symptomCount: number;
  backend: string;
  vectorStore: string;
  explainer: string;
  alpha: number;
  mode: string;
  // The server-emitted trace. When `streaming` is true this list grows
  // as SSE events arrive; otherwise it's the static post-complete trace.
  trace?: PipelineStage[];
  // Pre-fetched project insights (rule counts, source distribution, alpha
  // sweep). Used to render small contextual charts inside relevant stages.
  insights?: InsightsResponse | null;
  // True while we're still receiving SSE stage events. Drives the queued
  // walkthrough: stages animate as they arrive, and the walkthrough idles
  // if the next stage hasn't streamed yet.
  streaming?: boolean;
}

// While running we don't know the real stages yet. Show these placeholders
// that approximate what's about to happen; once trace arrives we replace.
const ESTIMATED_STAGES: { key: string; label: string; caption: string; est_ms: number }[] = [
  { key: "build_query",     label: "Build query string",   caption: "Symptoms to natural-language probe",                est_ms: 30 },
  { key: "expand_synonyms", label: "Synonym expansion",    caption: "UMLS-style clinical terms appended",               est_ms: 30 },
  { key: "encode",          label: "Encode query",         caption: "Azure OpenAI text-embedding-3-large to 3072d",      est_ms: 700 },
  { key: "vector_search",   label: "Vector search",        caption: "Pinecone 255-data-mining, top-30 cosine",         est_ms: 600 },
  { key: "attribute",       label: "Disease attribution",  caption: "Match passages to 41 Kaggle classes",              est_ms: 50 },
  { key: "related",         label: "Related context",      caption: "Top-5 nearby passages",                            est_ms: 200 },
  { key: "mining",          label: "Mining scorer",        caption: "FP-Growth: 23,839 rules",                          est_ms: 30 },
  { key: "fuse",            label: "Hybrid fusion",        caption: "Linear combine of mining + retrieval",             est_ms: 10 },
  { key: "evidence",        label: "Evidence cards",       caption: "Claim-level extraction, tier sort",               est_ms: 50 },
  { key: "explain",         label: "Clinical explanation", caption: "GPT-5.3, 4-section JSON, citations",             est_ms: 5000 },
];

// Per-stage visual budget for the auto-walkthrough. Most signature
// animations finish in ~1.5s; we add a hold so the viewer can read the
// final state before the next stage opens. The explain stage's typewriter
// runs longer so it gets a bigger budget.
const WALKTHROUGH_BUDGET_MS: Record<string, number> = {
  build_query: 2200,
  expand_synonyms: 2400,
  encode: 2600,
  vector_search: 2800,
  attribute: 2800,
  related: 2200,
  mining: 2400,
  fuse: 2800,
  cross_encoder: 2400,
  evidence: 2400,
  explain: 4500,
};
const WALKTHROUGH_DEFAULT_MS = 2200;

export function PipelineTimeline(p: Props) {
  const [activeIdx, setActiveIdx] = useState(-1);
  const [elapsed, setElapsed] = useState(0);
  // Multiple stages can be expanded at once now. Walkthrough pops them
  // open in sequence; manual click toggles individual stages.
  const [openKeys, setOpenKeys] = useState<Set<string>>(new Set());
  // Per-stage runId, bumps when a stage gets reopened so its animation
  // hooks replay. Indexed by stage key.
  const [openIds, setOpenIds] = useState<Record<string, number>>({});

  // Walkthrough state. `playingIdx` is the index in the trace of the stage
  // currently being played; -1 means walkthrough is not running. The
  // `enabled` flag persists across runs so users can opt out for repeat
  // diagnoses.
  const [walkEnabled, setWalkEnabled] = useState(true);
  const [playingIdx, setPlayingIdx] = useState(-1);
  // Track which stage keys we've already flashed the "just arrived" accent
  // on, so the animation fires exactly once per stage per request.
  const [arrivedKeys, setArrivedKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (p.state !== "running") return;
    setActiveIdx(0);
    setElapsed(0);
    // Fresh request, clear any previously expanded panels.
    setOpenKeys(new Set());
    setOpenIds({});
    setPlayingIdx(-1);
    setArrivedKeys(new Set());
    const start = performance.now();
    const ticker = setInterval(() => setElapsed(performance.now() - start), 80);

    let cumulative = 0;
    const timeouts: ReturnType<typeof setTimeout>[] = [];
    ESTIMATED_STAGES.forEach((s, i) => {
      cumulative += s.est_ms;
      timeouts.push(
        setTimeout(() => {
          if (i < ESTIMATED_STAGES.length - 1) setActiveIdx(i + 1);
        }, cumulative)
      );
    });
    return () => {
      clearInterval(ticker);
      timeouts.forEach(clearTimeout);
    };
  }, [p.state]);

  useEffect(() => {
    if (p.state === "done") setActiveIdx(ESTIMATED_STAGES.length);
  }, [p.state]);

  // ------------------------------------------------------------------
  // Queued walkthrough. Plays each stage's signature animation in
  // sequence, never starting stage N+1 until stage N's animation has
  // finished. If the server is faster than the animation budget the
  // queue depth grows; if slower, the walkthrough idles on stage N
  // waiting for stage N+1 to arrive over the stream.
  // ------------------------------------------------------------------
  // Trace + streaming refs so the advance loop reads the current values
  // without stale closures. Updated on every render; the loop reads them.
  const traceRef = useRef<PipelineStage[] | undefined>(p.trace);
  const streamingRef = useRef<boolean>(p.streaming ?? false);
  useEffect(() => {
    traceRef.current = p.trace;
  }, [p.trace]);
  useEffect(() => {
    streamingRef.current = p.streaming ?? false;
  }, [p.streaming]);

  // Track the most-recently-arrived stage key; the row gets a one-shot
  // accent flash. Cleared after the CSS animation completes (700ms +
  // a small buffer) so a refresh on the same key still fires.
  const [flashKey, setFlashKey] = useState<string | null>(null);
  useEffect(() => {
    if (!p.trace || p.trace.length === 0) return;
    const last = p.trace[p.trace.length - 1].key;
    if (arrivedKeys.has(last)) return;
    setArrivedKeys((cur) => {
      const next = new Set(cur);
      next.add(last);
      return next;
    });
    setFlashKey(last);
    const t = setTimeout(() => setFlashKey(null), 800);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.trace?.length]);

  // The walkthrough kicks off when:
  //   - we're streaming and at least one stage has arrived, OR
  //   - state went to 'done' with a non-empty trace.
  // Either way it runs exactly once per (state x walkEnabled) lifecycle.
  const walkArmed =
    walkEnabled &&
    ((p.streaming && (p.trace?.length ?? 0) > 0) || p.state === "done");
  useEffect(() => {
    if (!walkArmed) return;
    if ((traceRef.current?.length ?? 0) === 0) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setPlayingIdx(0);

    const advance = (i: number) => {
      if (cancelled) return;
      const trace = traceRef.current ?? [];
      if (i >= trace.length) {
        // No stage at index i yet. If the stream is still open, idle and
        // poll. If the stream is closed and we've played everything, exit.
        if (streamingRef.current) {
          timer = setTimeout(() => advance(i), 200);
          return;
        }
        setPlayingIdx(-1);
        return;
      }
      const stage = trace[i];
      setOpenKeys((cur) => {
        const next = new Set(cur);
        next.add(stage.key);
        return next;
      });
      setOpenIds((cur) => ({
        ...cur,
        [stage.key]: (cur[stage.key] ?? 0) + 1,
      }));
      setPlayingIdx(i);
      const budget =
        WALKTHROUGH_BUDGET_MS[stage.key] ?? WALKTHROUGH_DEFAULT_MS;
      timer = setTimeout(() => advance(i + 1), budget);
    };

    timer = setTimeout(() => advance(0), 250);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // We intentionally fire only when the walkthrough arms (or disarms),
    // not on every trace tick. The traceRef gives us the current list at
    // each advance() call without re-triggering the effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [walkArmed]);

  // If the user disables walkthrough mid-flight, bail out cleanly.
  useEffect(() => {
    if (!walkEnabled) setPlayingIdx(-1);
  }, [walkEnabled]);

  // Keep the currently-playing stage in view by scrolling its row into
  // the viewport whenever playingIdx advances. We bind the data-play
  // attribute as a query target.
  useEffect(() => {
    if (playingIdx < 0) return;
    const el = document.querySelector<HTMLElement>(
      `.timeline__item[data-play="playing"]`
    );
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [playingIdx]);

  if (p.state === "idle") return null;

  const isDone = p.state === "done";
  // Build the stage list. Three cases:
  //   - 'running' with no streamed trace yet to all estimates.
  //   - streaming or done with real trace to real stages first, then any
  //     remaining estimated rows for stages that haven't arrived yet.
  //     Lets the user see the full pipeline shape even mid-stream.
  const stages: {
    key: string;
    label: string;
    detail: string;
    ms?: number;
    data?: Record<string, any>;
  }[] = (() => {
    const realTrace = p.trace ?? [];
    if (realTrace.length === 0) {
      return ESTIMATED_STAGES.map((s) => ({
        key: s.key,
        label: s.label,
        detail: s.caption,
        ms: undefined,
      }));
    }
    const seen = new Set(realTrace.map((s) => s.key));
    const tail = ESTIMATED_STAGES.filter((s) => !seen.has(s.key)).map((s) => ({
      key: s.key,
      label: s.label,
      detail: s.caption,
      ms: undefined,
    }));
    return [...realTrace.map((s) => ({ ...s })), ...tail];
  })();

  const totalReal =
    p.trace && p.trace.length > 0
      ? p.trace.reduce((a, s) => a + (s.ms || 0), 0)
      : undefined;

  return (
    <div className="card timeline">
      <div className="timeline__head">
        <div>
          <div className="section-label" style={{ margin: 0 }}>
            Pipeline
            <span className="count">
              {isDone
                ? playingIdx >= 0 && p.trace
                  ? `walkthrough, stage ${playingIdx + 1} of ${p.trace.length}, real time ${(totalReal ?? 0).toFixed(0)}ms`
                  : `completed, ${(totalReal ?? 0).toFixed(0)}ms, click any stage to inspect`
                : `running, ${(elapsed / 1000).toFixed(1)}s`}
            </span>
          </div>
        </div>
        {/* Walkthrough controls, visible after a response lands */}
        {isDone && p.trace && p.trace.length > 0 && (
          <div className="timeline__controls">
            {playingIdx >= 0 ? (
              <button
                type="button"
                className="btn btn--ghost timeline__skip"
                onClick={() => setPlayingIdx(-1)}
                title="Stop the auto-walkthrough; all played stages stay open."
              >
                Skip walkthrough ▶▶
              </button>
            ) : (
              <label
                className="checkbox timeline__autoplay"
                title="When on, each new diagnose runs an auto-walkthrough that opens each stage in sequence."
              >
                <input
                  type="checkbox"
                  checked={walkEnabled}
                  onChange={(e) => setWalkEnabled(e.target.checked)}
                />
                Auto-walkthrough
              </label>
            )}
          </div>
        )}
        <div className="timeline__progress">
          <div
            className="timeline__progress-fill"
            style={{
              width: isDone
                ? playingIdx >= 0 && p.trace
                  ? `${Math.min(100, ((playingIdx + 1) / p.trace.length) * 100)}%`
                  : "100%"
                : `${Math.min(100, (activeIdx / ESTIMATED_STAGES.length) * 100 + 6)}%`,
            }}
          />
        </div>
      </div>

      <ol className="timeline__list">
        {stages.map((s, i) => {
          // Per-stage status. While streaming, anything we have real data
          // for is "done"; the next-up estimated row is "running"; tail
          // rows that haven't arrived yet are "queued".
          const hasRealData = !!s.data && Object.keys(s.data ?? {}).length > 0;
          const status = hasRealData
            ? "done"
            : isDone || i < activeIdx
            ? "done"
            : i === activeIdx
            ? "running"
            : "queued";
          const expandable = hasRealData;
          const open = openKeys.has(s.key);
          const stageRunId = openIds[s.key] ?? 0;
          // The currently-playing stage (during walkthrough) gets an
          // accent ring so the eye lands on it. Stages already played
          // stay highlighted as "done"; stages not yet visited dim.
          const playState =
            playingIdx < 0
              ? "idle"
              : i === playingIdx
              ? "playing"
              : i < playingIdx
              ? "played"
              : "pending";
          return (
            <li
              key={s.key}
              className="timeline__item"
              data-status={status}
              data-play={playState}
              data-just-arrived={flashKey === s.key ? "true" : "false"}
            >
              <span className="timeline__icon">
                {status === "done" && <Icon name="check" size={12} />}
                {status === "running" && <span className="spinner" />}
                {status === "queued" && <span className="dot-empty" />}
              </span>
              <div className="timeline__body">
                <div
                  className="timeline__row"
                  onClick={() => {
                    if (!expandable) return;
                    // Manual click cancels the walkthrough, the user has
                    // taken over.
                    setPlayingIdx(-1);
                    setOpenKeys((cur) => {
                      const next = new Set(cur);
                      if (next.has(s.key)) next.delete(s.key);
                      else next.add(s.key);
                      return next;
                    });
                    if (!open) {
                      setOpenIds((cur) => ({
                        ...cur,
                        [s.key]: (cur[s.key] ?? 0) + 1,
                      }));
                    }
                  }}
                  style={{ cursor: expandable ? "pointer" : "default" }}
                >
                  <span className="timeline__label">
                    {s.label}
                    {expandable && (
                      <span
                        className="timeline__chev"
                        data-open={open}
                        aria-hidden
                      >
                        <Icon name="chevron" size={11} />
                      </span>
                    )}
                  </span>
                  <span className="timeline__ms">
                    {s.ms !== undefined ? (
                      // When the row is open, count up the ms in sync with
                      // the signature animation. Otherwise just show the
                      // static number.
                      open ? (
                        <MsCounter target={s.ms} runId={stageRunId} />
                      ) : (
                        `${s.ms.toFixed(0)}ms`
                      )
                    ) : status === "running" ? (
                      "…"
                    ) : (
                      ""
                    )}
                  </span>
                </div>
                <div className="timeline__caption">{s.detail}</div>
                <AnimatePresence initial={false}>
                  {open && s.data && (
                    <motion.div
                      key="inspector"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{
                        duration: 0.32,
                        ease: [0.4, 0, 0.2, 1],
                      }}
                      style={{ overflow: "hidden" }}
                    >
                      <StageInspector
                        stageKey={s.key}
                        data={s.data}
                        ms={s.ms ?? 0}
                        runId={stageRunId}
                        insights={p.insights ?? null}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </li>
          );
        })}
      </ol>

      {!isDone && (
        <div className="timeline__foot">
          <Icon name="info" size={12} />
          <span>
            Showing live estimates while the request is in flight. Real
            per-stage timings and the data flowing through each stage will
            appear when the response lands. Click any stage to inspect.
          </span>
        </div>
      )}
    </div>
  );
}

// --- Per-stage inspector --------------------------------------------------
//
// Each stage's `data` payload has a different shape. We render a custom view
// per known stage; unknown stages fall back to a JSON dump.

function StageInspector({
  stageKey,
  data,
  ms,
  runId,
  insights,
}: {
  stageKey: string;
  data: Record<string, any>;
  ms: number;
  runId: number;
  insights: InsightsResponse | null;
}) {
  switch (stageKey) {
    case "build_query":
      return <BuildQueryInspector data={data} ms={ms} runId={runId} />;
    case "expand_synonyms":
      return <ExpandSynonymsInspector data={data} ms={ms} runId={runId} />;
    case "encode":
      return <EncodeInspector data={data} ms={ms} runId={runId} />;
    case "vector_search":
      return <VectorSearchInspector data={data} ms={ms} runId={runId} />;
    case "attribute":
      return (
        <AttributeInspector
          data={data}
          ms={ms}
          runId={runId}
          insights={insights}
        />
      );
    case "related":
      return <RelatedInspector data={data} ms={ms} runId={runId} />;
    case "mining":
      return (
        <MiningInspector
          data={data}
          ms={ms}
          runId={runId}
          insights={insights}
        />
      );
    case "fuse":
      return (
        <FuseInspector
          data={data}
          ms={ms}
          runId={runId}
          insights={insights}
        />
      );
    case "cross_encoder":
      return <CrossEncoderInspector data={data} ms={ms} runId={runId} />;
    case "evidence":
      return <EvidenceInspector data={data} ms={ms} runId={runId} />;
    case "explain":
      return (
        <ExplainInspector
          data={data}
          ms={ms}
          runId={runId}
          insights={insights}
        />
      );
    default:
      return (
        <div className="inspector">
          <pre className="inspector__json">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      );
  }
}

// --- Per-stage inspectors with signature animations -----------------------

// Build query: typewriter on the natural-language probe.
// Deliberately paced (~35 cps) so a demo viewer can read it forming.
function BuildQueryInspector({
  data,
  ms,
  runId,
}: {
  data: any;
  ms: number;
  runId: number;
}) {
  const typed = useTypewriter(data.query as string, 35, runId);
  const done = typed.length === (data.query as string).length;
  return (
    <div className="inspector">
      <StageNarration>
        Composed a {(data.query as string).length}-character probe in {fmtMs(ms)}.
      </StageNarration>
      <Field label="Encoded probe">
        <code className="inspector__code">
          {typed}
          <span
            className="caret"
            data-blink={done ? "true" : "false"}
            aria-hidden
          />
        </code>
      </Field>
    </div>
  );
}

// Synonym expansion: arrow draws from token to synonym, staggered.
function ExpandSynonymsInspector({
  data,
  ms,
  runId,
}: {
  data: any;
  ms: number;
  runId: number;
}) {
  const entries = Object.entries((data.applied ?? {}) as Record<string, string[]>);
  // Hooks must run in stable order, call before any early-return.
  const visible = useStaggeredReveal(entries.length, 160, runId);
  if (entries.length === 0) {
    return (
      <div className="inspector">
        <StageNarration>
          Synonym expansion was off, the probe passed through unchanged.
        </StageNarration>
      </div>
    );
  }
  const totalSyns = entries.reduce((a, [, v]) => a + v.length, 0);
  return (
    <div className="inspector">
      <StageNarration>
        Mapped {entries.length} symptom token{entries.length === 1 ? "" : "s"} to{" "}
        {totalSyns} clinical synonym{totalSyns === 1 ? "" : "s"} in {fmtMs(ms)} , 
        probe length grew by {data.delta_chars} chars.
      </StageNarration>
      <Field label="Mappings applied">
        <ul className="syn-list">
          {entries.map(([k, v], i) => (
            <li
              key={k}
              className="syn-row"
              data-revealed={i < visible}
              style={{ ["--i" as any]: i }}
            >
              <span className="syn-row__from mono">{k}</span>
              <span className="syn-row__arrow" aria-hidden>
                to
              </span>
              <span className="syn-row__to">{v.join(", ")}</span>
            </li>
          ))}
        </ul>
      </Field>
      <Field label="Final probe sent to encoder">
        <code className="inspector__code">{data.expanded_query}</code>
      </Field>
    </div>
  );
}

// Encode: count up the 8 vector chips from 0 to their final values.
function EncodeInspector({ data, ms, runId }: { data: any; ms: number; runId: number }) {
  const norm = useCountUp(Number(data.vector_norm), 1000, runId);
  const dim = useCountUp(Number(data.dim), 1000, runId);
  return (
    <div className="inspector">
      <StageNarration>
        {data.backend} returned a {data.dim}d unit vector in {fmtMs(ms)} (round-trip to Azure).
      </StageNarration>
      <div className="inspector__grid">
        <Field label="Backend">
          <span className="mono">{data.backend}</span>
        </Field>
        <Field label="Dimensions">
          <span className="mono">{Math.round(dim).toLocaleString()}</span>
        </Field>
        <Field label="L2 norm">
          <span className="mono">{norm.toFixed(4)}</span>
        </Field>
      </div>
      <Field label="Vector preview (first 8 dims)">
        <div className="inspector__vector">
          {(data.vector_preview as number[]).map((v, i) => (
            <VectorChip key={i} target={v} runId={runId} delay={i * 110} />
          ))}
        </div>
      </Field>
    </div>
  );
}
function VectorChip({
  target,
  runId,
  delay,
}: {
  target: number;
  runId: number;
  delay: number;
}) {
  // Stagger by re-keying on (runId, delay) so each chip starts at its
  // own offset, gives the cascade effect.
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    setArmed(false);
    const t = setTimeout(() => setArmed(true), delay);
    return () => clearTimeout(t);
  }, [runId, delay]);
  const v = useCountUp(armed ? target : 0, 650, `${runId}-${delay}`);
  return (
    <span className="inspector__dim" data-armed={armed}>
      {v >= 0 ? "+" : ""}
      {v.toFixed(4)}
    </span>
  );
}

// Vector search: top-10 matches slide in from the right with their scores.
function VectorSearchInspector({
  data,
  ms,
  runId,
}: {
  data: any;
  ms: number;
  runId: number;
}) {
  const matches = data.top_matches as any[];
  const visible = useStaggeredReveal(matches.length, 110, runId);
  const top = matches[0];
  return (
    <div className="inspector">
      <StageNarration>
        {data.store} returned top-{data.top_k} cosine matches in {fmtMs(ms)}.
        {top && <> Top hit: <em>{top.focus}</em> at {top.score.toFixed(4)} ({top.source}).</>}
      </StageNarration>
      <div className="inspector__grid">
        <Field label="Store">
          <span className="mono">{data.store}</span>
        </Field>
        <Field label="Top-K">
          <span className="mono">{data.top_k}</span>
        </Field>
        {data.filter && (
          <Field label="Filter">
            <span className="mono">{JSON.stringify(data.filter)}</span>
          </Field>
        )}
      </div>
      <Field label={`Top ${matches.length} matches`}>
        <table className="inspector__table inspector__table--anim">
          <thead>
            <tr>
              <th>#</th>
              <th>Score</th>
              <th>Source</th>
              <th>Focus</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((m, i) => (
              <tr
                key={m.passage_id}
                className="anim-row"
                data-revealed={i < visible}
                style={{ ["--i" as any]: i }}
              >
                <td className="mono">{i + 1}</td>
                <td className="mono">{m.score.toFixed(4)}</td>
                <td>{m.source}</td>
                <td>{m.focus}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Field>
      {matches.length > 1 && (
        <Field label="Cosine score fall-off, top-K">
          <div className="inspector__chart">
            <ResponsiveContainer width="100%" height={120}>
              <LineChart
                data={matches.map((m, i) => ({
                  rank: i + 1,
                  score: m.score,
                }))}
                margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="rank"
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 6, fontSize: 11, color: "var(--ink)",
                  }}
                />
                <Line
                  dataKey="score"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="inspector__chart-caption">
            A steep fall-off means the query has a clear winner; a flat curve
            means the top-K is ambiguous (good signal that fusion will need to
            do real work).
          </p>
        </Field>
      )}
    </div>
  );
}

// Disease attribution: bar chart races. Each bar fills from 0 to its score.
function AttributeInspector({
  data,
  ms,
  runId,
  insights,
}: {
  data: any;
  ms: number;
  runId: number;
  insights: InsightsResponse | null;
}) {
  const entries = Object.entries(
    data.per_disease_top_score as Record<string, number>
  );
  const sample = data.sample as any[];
  const sampleVisible = useStaggeredReveal(sample.length, 130, runId);
  // Build a source-distribution chart restricted to sources that actually
  // contributed at least one passage to the top-30. We pull the counts
  // from /insights and intersect with the sources we observed in this
  // request's sample. (Truthfully, the trace only carries 8 sample rows
  // but the full set is in data.top_matches up the stack, for now, just
  // intersect with the unique sources visible.)
  const sourceCounts = (insights?.source_distribution ?? []).slice(0, 6);
  return (
    <div className="inspector">
      <StageNarration>
        Mapped {sample.length}+ raw passages to {data.diseases_hit} of{" "}
        {data.universe_size} disease classes in {fmtMs(ms)} via curated keyword
        matching on each passage's focus/question.
      </StageNarration>
      <div className="inspector__grid">
        <Field label="Diseases hit">
          <span className="mono">
            {data.diseases_hit} / {data.universe_size}
          </span>
        </Field>
      </div>
      <Field label="Per-disease top score">
        <div className="inspector__bars">
          {entries.map(([d, s], i) => (
            <RaceBar
              key={d}
              label={d.replace(/_/g, " ")}
              target={s}
              runId={runId}
              delay={i * 180}
            />
          ))}
        </div>
      </Field>
      <Field label="Sample attributions">
        <table className="inspector__table inspector__table--anim">
          <thead>
            <tr>
              <th>Passage</th>
              <th>Score</th>
              <th>Mapped to</th>
            </tr>
          </thead>
          <tbody>
            {sample.map((s, i) => (
              <tr
                key={s.passage_id}
                className="anim-row"
                data-revealed={i < sampleVisible}
                style={{ ["--i" as any]: i }}
              >
                <td className="mono">{s.passage_id.slice(0, 8)}…</td>
                <td className="mono">{s.score.toFixed(4)}</td>
                <td>{s.matched_diseases.join(", ").replace(/_/g, " ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Field>
      {sourceCounts.length > 0 && (
        <Field label="Where MedQuAD passages come from (overall corpus)">
          <div className="inspector__chart">
            <ResponsiveContainer width="100%" height={140}>
              <BarChart
                data={sourceCounts}
                margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  interval={0}
                />
                <YAxis
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 6, fontSize: 11, color: "var(--ink)",
                  }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {sourceCounts.map((s, i) => (
                    <Cell key={i} fill={TIER_COLORS[s.tier - 1] ?? "#94a3b8"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Field>
      )}
    </div>
  );
}
function RaceBar({
  label,
  target,
  runId,
  delay,
}: {
  label: string;
  target: number;
  runId: number;
  delay: number;
}) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    setArmed(false);
    const t = setTimeout(() => setArmed(true), delay);
    return () => clearTimeout(t);
  }, [runId, delay]);
  const value = useCountUp(armed ? target : 0, 1100, `${runId}-${delay}`);
  return (
    <div className="inspector__bar-row">
      <span className="inspector__bar-label">{label}</span>
      <div className="inspector__bar">
        <div
          className="inspector__bar-fill"
          style={{
            width: `${Math.min(100, value * 100)}%`,
            transition: "width 0.05s linear",
          }}
        />
      </div>
      <span className="mono inspector__bar-num">{value.toFixed(3)}</span>
    </div>
  );
}

// Related: passage cards fade in.
function RelatedInspector({ data, ms, runId }: { data: any; ms: number; runId: number }) {
  const items = data.items as any[];
  const visible = useStaggeredReveal(items.length, 140, runId);
  return (
    <div className="inspector">
      <StageNarration>
        Pulled {items.length} nearby passages in {fmtMs(ms)} for the "Related context"
        rail at the bottom of the page, these aren't required to map to a disease.
      </StageNarration>
      <Field label="Top-5 nearby passages">
        <table className="inspector__table inspector__table--anim">
          <thead>
            <tr>
              <th>Score</th>
              <th>Source</th>
              <th>Focus</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr
                key={i}
                className="anim-row"
                data-revealed={i < visible}
                style={{ ["--i" as any]: i }}
              >
                <td className="mono">{it.score.toFixed(4)}</td>
                <td>{it.source}</td>
                <td>{it.focus}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Field>
    </div>
  );
}

// Mining: fired rules slide in, antecedent in mono pill emphasized.
function MiningInspector({
  data,
  ms,
  runId,
  insights,
}: {
  data: any;
  ms: number;
  runId: number;
  insights: InsightsResponse | null;
}) {
  const rules = data.top_fired as any[];
  const visible = useStaggeredReveal(rules.length, 160, runId);
  // For each disease that fired, look up its total rule count in the
  // mined-rules table (top-25 only, so coverage is 'best-effort'). Lets
  // the demo viewer see WHY a disease scored well, strong rule coverage.
  const firedDiseaseSlugs = new Set(
    rules.map((r) => (r.disease as string).replace(/_/g, " "))
  );
  const ruleCoverage = (insights?.rules_per_disease ?? [])
    .filter((r) => firedDiseaseSlugs.has(r.disease))
    .slice(0, 5);
  return (
    <div className="inspector">
      <StageNarration>
        Looked up {data.diseases_scored} of 41 diseases against{" "}
        {(data.rules_in_table as number).toLocaleString()} mined rules in{" "}
        {fmtMs(ms)}. Each disease's score is{" "}
        <code>max(confidence × overlap)</code> over rules whose antecedent ⊆ Q.
      </StageNarration>
      <div className="inspector__grid">
        <Field label="Diseases scored">
          <span className="mono">{data.diseases_scored}</span>
        </Field>
        <Field label="Rules in table">
          <span className="mono">
            {(data.rules_in_table as number).toLocaleString()}
          </span>
        </Field>
      </div>
      <Field label="Top fired rules">
        {rules.length === 0 ? (
          <p className="inspector__muted">No rules fired for this query.</p>
        ) : (
          <ul className="rule-list">
            {rules.map((r, i) => (
              <li
                key={i}
                className="rule-row"
                data-revealed={i < visible}
                style={{ ["--i" as any]: i }}
              >
                <span className="rule-row__ante mono">{`{${r.antecedent.join(", ")}}`}</span>
                <span className="rule-row__arrow" aria-hidden>
                  to
                </span>
                <span className="rule-row__disease">
                  {r.disease.replace(/_/g, " ")}
                </span>
                <span className="rule-row__stats mono">
                  conf {r.confidence.toFixed(2)}, lift {r.lift.toFixed(1)} , 
                  score {r.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Field>
      {ruleCoverage.length > 0 && (
        <Field label="Rule coverage for the diseases that fired (mined table)">
          <div className="inspector__chart">
            <ResponsiveContainer width="100%" height={120}>
              <BarChart
                data={ruleCoverage}
                layout="vertical"
                margin={{ top: 4, right: 16, left: 70, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <YAxis
                  type="category"
                  dataKey="disease"
                  width={70}
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 6, fontSize: 11, color: "var(--ink)",
                  }}
                />
                <Bar
                  dataKey="rules"
                  fill="var(--accent)"
                  radius={[0, 3, 3, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="inspector__chart-caption">
            More rules ≠ better, but a disease with no rules can't fire here.
            Coverage skew is one reason the headline metric is fused, not
            mining-only.
          </p>
        </Field>
      )}
    </div>
  );
}

// Fuse: visualize α-blend. Mining + retrieval bars, fused number counts up.
function FuseInspector({
  data,
  ms,
  runId,
  insights,
}: {
  data: any;
  ms: number;
  runId: number;
  insights: InsightsResponse | null;
}) {
  const candidates = data.candidates as any[];
  const visible = useStaggeredReveal(candidates.length, 220, runId);
  return (
    <div className="inspector">
      <StageNarration>
        Combined the two signals via{" "}
        <code>FusedScore = α, retrieval + (1-α), mining</code> with α=
        {Number(data.alpha).toFixed(2)} in {fmtMs(ms)}. Top match:{" "}
        <em>{candidates[0]?.disease.replace(/_/g, " ")}</em> at{" "}
        {candidates[0]?.fused_score.toFixed(3)}.
      </StageNarration>
      <div className="inspector__grid">
        <Field label="α">
          <span className="mono">{Number(data.alpha).toFixed(2)}</span>
        </Field>
        <Field label="Mode">
          <span className="mono">{data.mode}</span>
        </Field>
      </div>
      <Field label="Score breakdown">
        <ul className="fuse-list">
          {candidates.map((c, i) => (
            <FuseRow
              key={c.disease}
              c={c}
              i={i}
              revealed={i < visible}
              runId={runId}
            />
          ))}
        </ul>
      </Field>
      <Field label="This case, ranking under different α settings">
        <div className="inspector__chart">
          <table className="inspector__table" style={{ marginTop: 0 }}>
            <thead>
              <tr>
                <th>α setting</th>
                <th>Top-1 disease</th>
                <th className="num">Top-1 score</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody>
              {[
                { a: 0.0, label: "0.00, mining-only" },
                { a: 0.3, label: "0.30, fused (default)" },
                { a: 0.5, label: "0.50, balanced" },
                { a: 1.0, label: "1.00, retrieval-only" },
              ].map(({ a, label }) => {
                const ranked = [...candidates]
                  .map((c: any) => ({
                    disease: c.disease,
                    s: a * c.retrieval_score + (1 - a) * c.mining_score,
                  }))
                  .sort((x, y) => y.s - x.s);
                const top = ranked[0];
                const isCurrent = Math.abs(a - Number(data.alpha)) < 0.01;
                return (
                  <tr key={a} style={isCurrent ? { fontWeight: 500 } : {}}>
                    <td className="mono">{label}</td>
                    <td>{top.disease.replace(/_/g, " ")}</td>
                    <td className="num mono">{top.s.toFixed(3)}</td>
                    <td className="mono">
                      {a === 0 ? "mining" : a === 1 ? "retrieval" : "fused"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="inspector__chart-caption">
          Same query re-fused under four α points without re-running the
          pipeline. If the top-1 disease changes across rows, fusion is
          load-bearing for this case.
        </p>
      </Field>
      {insights && insights.alpha_sweep.length > 0 && (
        <Field label="α-sweep, why we ship α=0.30">
          <div className="inspector__chart">
            <ResponsiveContainer width="100%" height={150}>
              <LineChart
                data={insights.alpha_sweep}
                margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="alpha"
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 6, fontSize: 11, color: "var(--ink)",
                  }}
                />
                <ReferenceLine
                  x={Number(data.alpha)}
                  stroke="var(--accent)"
                  strokeDasharray="4 4"
                />
                <Line
                  dataKey="recall@1"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  dataKey="mrr"
                  stroke="#c2772c"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="inspector__chart-caption">
            Recall@1 (blue) and MRR (orange) plateau on α∈[0.1, 0.4]. We
            operate at α=0.30, marked. Beyond ~0.6 the metrics collapse as
            retrieval starts dominating.
          </p>
        </Field>
      )}
    </div>
  );
}
function FuseRow({
  c,
  i,
  revealed,
  runId,
}: {
  c: any;
  i: number;
  revealed: boolean;
  runId: number;
}) {
  // Per-row animation: animate mining and retrieval bars after the row
  // becomes revealed; fused number counts up after a short hold so the
  // user perceives "mining + retrieval combine into fused".
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    setArmed(false);
    if (!revealed) return;
    const t = setTimeout(() => setArmed(true), 120);
    return () => clearTimeout(t);
  }, [revealed, runId]);
  const fused = useCountUp(armed ? c.fused_score : 0, 900, `${runId}-${i}`);
  return (
    <li
      className="fuse-row"
      data-revealed={revealed}
      style={{ ["--i" as any]: i }}
    >
      <span className="fuse-row__rank mono">{String(i + 1).padStart(2, "0")}</span>
      <span className="fuse-row__name">{c.disease.replace(/_/g, " ")}</span>
      <div className="fuse-row__bars">
        <span className="fuse-row__band">
          <span className="fuse-row__band-label">mining</span>
          <span className="fuse-row__bar fuse-row__bar--mining">
            <span
              className="fuse-row__bar-fill"
              style={{ width: armed ? `${c.mining_score * 100}%` : "0%" }}
            />
          </span>
          <span className="mono fuse-row__num">{c.mining_score.toFixed(3)}</span>
        </span>
        <span className="fuse-row__band">
          <span className="fuse-row__band-label">retrieval</span>
          <span className="fuse-row__bar fuse-row__bar--retrieval">
            <span
              className="fuse-row__bar-fill"
              style={{ width: armed ? `${c.retrieval_score * 100}%` : "0%" }}
            />
          </span>
          <span className="mono fuse-row__num">{c.retrieval_score.toFixed(3)}</span>
        </span>
      </div>
      <span className="fuse-row__fused mono">
        {fused.toFixed(3)}
        <small>fused</small>
      </span>
    </li>
  );
}

function CrossEncoderInspector({
  data,
  ms,
  runId,
}: {
  data: any;
  ms: number;
  runId: number;
}) {
  const cands = data.candidates as any[];
  const visible = useStaggeredReveal(cands.length, 160, runId);
  return (
    <div className="inspector">
      <StageNarration>
        Reranked the top-{cands.length} candidates' passages with {data.reranker} in {fmtMs(ms)}.
      </StageNarration>
      <Field label="Reranker">
        <span className="mono">{data.reranker}</span>
      </Field>
      <Field label="Rerank summary">
        <table className="inspector__table inspector__table--anim">
          <thead>
            <tr>
              <th>Disease</th>
              <th>K</th>
              <th>Top before</th>
              <th>Top after</th>
            </tr>
          </thead>
          <tbody>
            {cands.map((c, i) => (
              <tr
                key={i}
                className="anim-row"
                data-revealed={i < visible}
                style={{ ["--i" as any]: i }}
              >
                <td>{c.disease.replace(/_/g, " ")}</td>
                <td className="mono">{c.k}</td>
                <td className="mono">{c.top_before?.slice(0, 8) ?? ", "}…</td>
                <td className="mono">{c.top_after?.slice(0, 8) ?? ", "}…</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Field>
    </div>
  );
}

// Evidence: tier badges pop in, specificity bars fill.
function EvidenceInspector({ data, ms, runId }: { data: any; ms: number; runId: number }) {
  const rows = data.per_diagnosis as any[];
  const visible = useStaggeredReveal(rows.length, 160, runId);
  const totalCards = rows.reduce((a, r) => a + r.n_cards, 0);
  // Source distribution among selected top cards. Roll up per-source
  // count + tier so the chart can colour each bar correctly.
  const sourceTally = new Map<string, { count: number; tier: number }>();
  rows.forEach((r) => {
    if (!r.top_card) return;
    const src = r.top_card.source as string;
    const tier = r.top_card.tier as number;
    const cur = sourceTally.get(src);
    if (cur) cur.count += 1;
    else sourceTally.set(src, { count: 1, tier });
  });
  const sourceChartData = Array.from(sourceTally.entries()).map(
    ([label, { count, tier }]) => ({ label, count, tier })
  );
  return (
    <div className="inspector">
      <StageNarration>
        Built {totalCards} claim-level evidence card{totalCards === 1 ? "" : "s"} across{" "}
        {rows.length} diagnoses in {fmtMs(ms)}, sorted by{" "}
        (source tier ascending, specificity descending).
      </StageNarration>
      <Field label="Evidence cards per diagnosis">
        <table className="inspector__table inspector__table--anim">
          <thead>
            <tr>
              <th>Disease</th>
              <th>Cards</th>
              <th>Top source</th>
              <th>Tier</th>
              <th>Type</th>
              <th>Spec.</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <tr
                key={i}
                className="anim-row"
                data-revealed={i < visible}
                style={{ ["--i" as any]: i }}
              >
                <td>{d.disease.replace(/_/g, " ")}</td>
                <td className="mono">{d.n_cards}</td>
                <td>{d.top_card?.source ?? ", "}</td>
                <td>
                  {d.top_card ? (
                    <span className={`pill pill--tier${d.top_card.tier}`}>
                      <span className="dot" />t{d.top_card.tier}
                    </span>
                  ) : (
                    ", "
                  )}
                </td>
                <td>{d.top_card?.passage_type ?? ", "}</td>
                <td className="mono">
                  {d.top_card ? d.top_card.specificity.toFixed(2) : ", "}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Field>
      {sourceChartData.length > 0 && (
        <Field label="Source distribution, selected top cards">
          <div className="inspector__chart">
            <ResponsiveContainer width="100%" height={130}>
              <BarChart
                data={sourceChartData}
                margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  interval={0}
                />
                <YAxis
                  tick={{ fill: "var(--ink-3)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 6, fontSize: 11, color: "var(--ink)",
                  }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {sourceChartData.map((s, i) => (
                    <Cell key={i} fill={TIER_COLORS[s.tier - 1] ?? "#94a3b8"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="inspector__chart-caption">
            Where the top evidence card per diagnosis was sourced. Tier-1
            (NIH NHLBI / NIDDK / MedlinePlus / GHR / GARD / NINDS / CDC /
            NCI) is what we trust most.
          </p>
        </Field>
      )}
    </div>
  );
}

// Explain: typewrite the first preview, fade the rest in.
function ExplainInspector({
  data,
  ms,
  runId,
  insights,
}: {
  data: any;
  ms: number;
  runId: number;
  insights: InsightsResponse | null;
}) {
  const rows = data.per_diagnosis as any[];
  const firstPreview = rows[0]?.preview ?? "";
  const typed = useTypewriter(firstPreview, 45, runId);
  const visible = useStaggeredReveal(rows.length, 350, runId);
  const isLLM = String(data.backend).startsWith("openai");
  return (
    <div className="inspector">
      <StageNarration>
        {isLLM ? "Synthesised" : "Stitched"} {rows.length} four-section structured
        explanation{rows.length === 1 ? "" : "s"} in {fmtMs(ms)} via{" "}
        <code>{data.backend}</code>{data.model && <code>{` (${data.model})`}</code>}.
        Citations are appended deterministically.
      </StageNarration>
      <div className="inspector__grid">
        <Field label="Backend">
          <span className="mono">{data.backend}</span>
        </Field>
        {data.model && (
          <Field label="Model">
            <span className="mono">{data.model}</span>
          </Field>
        )}
      </div>
      <Field label="Per-diagnosis preview">
        <div className="inspector__previews">
          {rows.map((d, i) => (
            <div
              key={i}
              className="inspector__preview"
              data-revealed={i < visible}
              style={{ ["--i" as any]: i }}
            >
              <div className="inspector__preview-head">
                <span className="mono">{d.disease.replace(/_/g, " ")}</span>
                <span className="pill pill--mono">{d.backend}</span>
              </div>
              <p className="inspector__preview-text">
                {i === 0 ? typed : d.preview}
                {i === 0 && typed.length < firstPreview.length && (
                  <span className="caret" aria-hidden />
                )}
              </p>
              <span className="mono inspector__muted-sm">
                {d.citation_count} citation{d.citation_count === 1 ? "" : "s"}
              </span>
            </div>
          ))}
        </div>
      </Field>
      {insights && insights.headline_latency.length > 0 && (
        <Field label="This run vs. headline benchmark">
          <table className="inspector__table" style={{ marginTop: 0 }}>
            <thead>
              <tr>
                <th>Stage</th>
                <th className="num">benchmark mean</th>
                <th className="num">benchmark p95</th>
                <th className="num">this run</th>
              </tr>
            </thead>
            <tbody>
              {insights.headline_latency.map((b) => {
                // The "explain" benchmark stage in the CSV is the
                // template explainer; this run's ms (LLM) will dwarf it.
                // We mark dramatic deltas honestly.
                const thisRun =
                  b.stage === "explain"
                    ? ms
                    : b.stage === "TOTAL"
                    ? undefined  // we don't surface total here
                    : undefined;
                if (b.stage === "TOTAL") return null;
                return (
                  <tr key={b.stage}>
                    <td>{b.stage}</td>
                    <td className="num mono">{b.mean_ms.toFixed(2)}ms</td>
                    <td className="num mono">{b.p95_ms.toFixed(2)}ms</td>
                    <td
                      className="num mono"
                      style={
                        thisRun !== undefined
                          ? { color: "var(--accent-strong)" }
                          : { color: "var(--ink-4)" }
                      }
                    >
                      {thisRun !== undefined ? `${thisRun.toFixed(0)}ms` : ", "}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="inspector__chart-caption">
            Headline benchmark uses MiniLM + FAISS + the deterministic
            template explainer (n=100 queries). When the explainer is the
            OpenAI structured one (especially against gpt-5-class
            reasoning models), the explain stage is 50-100× the template
            baseline, that's the cost of going from stitch to LLM.
          </p>
        </Field>
      )}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="inspector__field">
      <div className="inspector__field-label">{label}</div>
      <div className="inspector__field-body">{children}</div>
    </div>
  );
}

// Counts up the right-margin ms when its parent row opens, in roughly the
// same wall-clock window as the signature animation. The duration is
// proportional to the value so a 700ms stage takes longer to count up
// than a 1ms stage, the eye reads "this took longer" naturally.
function MsCounter({
  target,
  runId,
}: {
  target: number;
  runId: number;
}) {
  // Cap visual duration at 1.4s so even a 30,000ms LLM call doesn't
  // count up forever. Below 50ms, just snap to the value (animating
  // those is too fast to read anyway).
  const visualMs = Math.min(1400, Math.max(0, target * 1.6));
  const value = useCountUp(target, target < 50 ? 0 : visualMs, runId);
  return <>{`${value.toFixed(0)}ms`}</>;
}

// One-line "narration" rendered at the top of each inspector. Calls out
// what this stage just did using the concrete numbers from `data`.
function StageNarration({ children }: { children: React.ReactNode }) {
  return (
    <div className="inspector__narration">
      <span className="inspector__narration-dot" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

function fmtMs(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}
