"use client";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Link from "next/link";
import { api, InsightsResponse } from "../api";
import { Icon } from "../components/Icon";
import { AuthGuard } from "../components/AuthGuard";
import { UserMenu } from "../components/UserMenu";

const TIER_COLORS = ["#2f7a4f", "#b88500", "#94a3b8"]; // tier 1 / 2 / 3

export default function InsightsPage() {
  return (
    <AuthGuard>
      <InsightsContent />
    </AuthGuard>
  );
}

function InsightsContent() {
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  async function refresh() {
    try {
      const r = await api.insights();
      setData(r);
      setErr(null);
    } catch (e: any) {
      setErr(e.message);
    }
  }
  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <div className="topbar__logo">R</div>
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
          <Link
            href="/insights"
            className="topbar__navlink topbar__navlink--active"
          >
            Insights
          </Link>
          <Link href="/profile" className="topbar__navlink">
            Profile
          </Link>
        </nav>
        <div className="topbar__spacer" />
        <button
          className="btn"
          onClick={refresh}
          style={{ fontSize: 12 }}
        >
          Refresh
        </button>
        <button
          className="icon-btn"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Toggle theme"
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={14} />
        </button>
        <UserMenu />
      </header>

      <main className="insights">
        {err && (
          <div className="error-banner">Could not load insights: {err}</div>
        )}
        {!data && !err && (
          <p style={{ color: "var(--ink-3)", fontSize: 13 }}>
            Loading aggregates from <code>data/results/*.csv</code>…
          </p>
        )}

        {data && (
          <>
            <h2 className="h-section">
              Project insights
              <span>
                4 charts, live latency from the last{" "}
                {data.latency_recent.length || 0} diagnoses
              </span>
            </h2>

            <div className="insights__grid">
              <AlphaSweepCard data={data} />
              <RulesPerDiseaseCard data={data} />
              <SourceDistributionCard data={data} />
              <LiveLatencyCard data={data} />
            </div>

            {data.headline_latency.length > 0 && (
              <HeadlineLatencyCard data={data} />
            )}
            {data.ablation.length > 0 && <AblationCard data={data} />}
          </>
        )}
      </main>
    </div>
  );
}

// --- Cards ----------------------------------------------------------------

function CardHeader({
  title,
  caption,
}: {
  title: string;
  caption: string;
}) {
  return (
    <div className="insights__head">
      <h3 className="insights__title">{title}</h3>
      <p className="insights__caption">{caption}</p>
    </div>
  );
}

function AlphaSweepCard({ data }: { data: InsightsResponse }) {
  if (data.alpha_sweep.length === 0) return null;
  // Mark the operating point at 0.3 in the chart
  return (
    <div className="card insights__card">
      <CardHeader
        title="α-sweep, Recall@K and MRR vs fusion weight"
        caption="0.0 = mining-only; 1.0 = retrieval-only. The default α=0.30 sits in the middle of the optimal plateau."
      />
      <ResponsiveContainer width="100%" height={260}>
        <LineChart
          data={data.alpha_sweep}
          margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="alpha"
            tick={{ fill: "var(--ink-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "var(--ink-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--ink)",
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine
            x={0.3}
            stroke="var(--accent)"
            strokeDasharray="4 4"
            label={{
              value: "α=0.30 default",
              position: "top",
              fill: "var(--accent-strong)",
              fontSize: 10,
            }}
          />
          <Line
            dataKey="recall@1"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
          <Line
            dataKey="recall@10"
            stroke="#7a9cc6"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
          <Line
            dataKey="mrr"
            stroke="#c2772c"
            strokeWidth={2}
            dot={{ r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function RulesPerDiseaseCard({ data }: { data: InsightsResponse }) {
  if (data.rules_per_disease.length === 0) return null;
  return (
    <div className="card insights__card">
      <CardHeader
        title="FP-Growth rules per disease (top 25)"
        caption="At min_support=0.005, min_confidence=0.5: 23,839 rules across 41 diseases. Coverage skews to high-prevalence classes."
      />
      <ResponsiveContainer width="100%" height={420}>
        <BarChart
          data={data.rules_per_disease}
          layout="vertical"
          margin={{ top: 8, right: 16, left: 110, bottom: 0 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            type="number"
            tick={{ fill: "var(--ink-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            type="category"
            dataKey="disease"
            tick={{ fill: "var(--ink-3)", fontSize: 10 }}
            width={110}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--ink)",
            }}
          />
          <Bar dataKey="rules" fill="var(--accent)" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SourceDistributionCard({ data }: { data: InsightsResponse }) {
  if (data.source_distribution.length === 0) return null;
  // Sort by count desc; top 12 + "Other"
  const sorted = [...data.source_distribution].sort((a, b) => b.count - a.count);
  const top = sorted.slice(0, 12);
  return (
    <div className="card insights__card">
      <CardHeader
        title="MedQuAD source distribution"
        caption="24,063 passages across 12 NIH source folders, colored by authority tier (1 = highest)."
      />
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={top}
          margin={{ top: 8, right: 8, left: -8, bottom: 60 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--ink-3)", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval={0}
            angle={-30}
            textAnchor="end"
          />
          <YAxis
            tick={{ fill: "var(--ink-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--ink)",
            }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {top.map((s, i) => (
              <Cell key={i} fill={TIER_COLORS[s.tier - 1] ?? "#94a3b8"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="insights__legend">
        <span>
          <span className="dot" style={{ background: TIER_COLORS[0] }} /> Tier 1
        </span>
        <span>
          <span className="dot" style={{ background: TIER_COLORS[1] }} /> Tier 2
        </span>
        <span>
          <span className="dot" style={{ background: TIER_COLORS[2] }} /> Tier 3
        </span>
      </div>
    </div>
  );
}

function LiveLatencyCard({ data }: { data: InsightsResponse }) {
  const recent = data.latency_recent.slice(-30);
  return (
    <div className="card insights__card">
      <CardHeader
        title="Live latency, last 30 diagnoses"
        caption={
          recent.length === 0
            ? "No diagnoses run yet in this API session. Run one on the Diagnose tab."
            : `Stacked breakdown: encode + vector search + mining + explain. Total = sum.`
        }
      />
      {recent.length > 0 && (
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={recent.map((r, i) => ({ ...r, idx: i + 1 }))}
            margin={{ top: 8, right: 8, left: -8, bottom: 0 }}
          >
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              dataKey="idx"
              tick={{ fill: "var(--ink-3)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
            />
            <YAxis
              tick={{ fill: "var(--ink-3)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{ value: "ms", angle: -90, position: "insideLeft",
                       fill: "var(--ink-4)", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border-strong)",
                borderRadius: 6,
                fontSize: 12,
                color: "var(--ink)",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar
              dataKey="encode_ms"
              stackId="t"
              fill="#3a86ff"
              name="encode"
            />
            <Bar
              dataKey="vector_search_ms"
              stackId="t"
              fill="#74b9ff"
              name="vector_search"
            />
            <Bar
              dataKey="mining_ms"
              stackId="t"
              fill="#b59dff"
              name="mining"
            />
            <Bar
              dataKey="explain_ms"
              stackId="t"
              fill="#f0a460"
              name="explain"
            />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function HeadlineLatencyCard({ data }: { data: InsightsResponse }) {
  return (
    <div className="card insights__card insights__card--full">
      <CardHeader
        title="Headline latency benchmark"
        caption="From data/results/latency_summary.csv. Default config: MiniLM + template explainer + FAISS, 100 queries on M3 Pro."
      />
      <table className="insights__table">
        <thead>
          <tr>
            <th>Stage</th>
            <th className="num">n</th>
            <th className="num">mean ms</th>
            <th className="num">p50 ms</th>
            <th className="num">p95 ms</th>
            <th className="num">max ms</th>
          </tr>
        </thead>
        <tbody>
          {data.headline_latency.map((r) => (
            <tr key={r.stage}>
              <td>{r.stage}</td>
              <td className="num mono">{r.n}</td>
              <td className="num mono">{r.mean_ms}</td>
              <td className="num mono">{r.p50_ms}</td>
              <td className="num mono">{r.p95_ms}</td>
              <td className="num mono">{r.max_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AblationCard({ data }: { data: InsightsResponse }) {
  return (
    <div className="card insights__card insights__card--full">
      <CardHeader
        title="Ablation summary"
        caption="From data/results/ablation_summary.csv. 200 synthetic test cases, α=0.3 for fused rows."
      />
      <table className="insights__table">
        <thead>
          <tr>
            <th>Variant</th>
            <th>Mode</th>
            <th className="num">R@1</th>
            <th className="num">R@3</th>
            <th className="num">R@5</th>
            <th className="num">R@10</th>
            <th className="num">MRR</th>
          </tr>
        </thead>
        <tbody>
          {data.ablation.map((r, i) => (
            <tr key={i}>
              <td>{r.variant}</td>
              <td>{r.mode}</td>
              <td className="num mono">{Number(r["recall@1"]).toFixed(3)}</td>
              <td className="num mono">{Number(r["recall@3"]).toFixed(3)}</td>
              <td className="num mono">{Number(r["recall@5"]).toFixed(3)}</td>
              <td className="num mono">{Number(r["recall@10"]).toFixed(3)}</td>
              <td className="num mono">{Number(r.mrr).toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
