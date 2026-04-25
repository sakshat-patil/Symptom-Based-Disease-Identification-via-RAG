// Tiny API client for the FastAPI service.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8001";

export interface ConfigDTO {
  backends: string[];
  explainers: string[];
  vector_store: string;
  metadata_filter_supported: boolean;
  openai_available: boolean;
  default_alpha: number;
}

export interface SourceDTO {
  source_id: string;
  label: string;
  tier: number;
  count: number;
}

export interface MatchingRule {
  antecedent: string[];
  confidence: number;
  lift: number;
  size: number;
}

export interface EvidenceClaim {
  sentence: string;
  matched_terms: string[];
  char_start: number;
  char_end: number;
}

export interface EvidenceCard {
  passage_id: string;
  source_id: string;
  source_label: string;
  source_tier: number;
  passage_type: string;
  focus: string;
  question: string;
  full_text: string;
  claims: EvidenceClaim[];
  specificity: number;
  retrieval_score: number;
  ce_score?: number | null;
}

export interface Diagnosis {
  disease: string;
  disease_pretty: string;
  fused_score: number;
  mining_score: number;
  retrieval_score: number;
  matching_rules: MatchingRule[];
  evidence_cards: EvidenceCard[];
  explanation: {
    symptom_disease_link: string;
    statistical_prior: string;
    evidence_quality: string;
    whats_missing: string;
    citations: string[];
    backend: string;
  };
}

export interface RelatedContext {
  passage_id: string;
  source: string;
  source_label: string;
  focus: string;
  question: string;
  text: string;
  score: number;
}

export interface PipelineStage {
  key: string;
  label: string;
  detail: string;
  ms: number;
  // Loose: each stage's payload has a different shape. The component
  // renders generically.
  data: Record<string, any>;
}

export interface DiagnoseResponse {
  query_symptoms: string[];
  used_alpha: number;
  used_backend: string;
  used_mode: string;
  explainer_backend: string;
  vector_store: string;
  diagnoses: Diagnosis[];
  related_context: RelatedContext[];
  latency_ms: Record<string, number>;
  pipeline_trace: PipelineStage[];
}

export interface DiagnoseRequest {
  symptoms: string[];
  backend: string;
  mode: string;
  alpha: number;
  expand_synonyms: boolean;
  cross_encoder: boolean;
  explainer: string;
  source_filter?: string | null;
  passage_type_filter?: string | null;
  top_n: number;
  // Optional retrieval knobs. Defaults match the headline numbers in
  // the report; the rail's "Tuning" panel exposes sliders.
  top_k_retrieval?: number;
  related_top_k?: number;
  max_evidence_cards?: number;
  trace?: boolean;
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

export interface SuggestResponse {
  suggestions: { symptom: string; score: number; n_rules: number }[];
}

export interface ExplainSymptomResponse {
  symptom: string;
  pretty: string;
  synonyms: string[];
  explanation: string;
  backend: string;
  top_diseases: string[];
}

export interface DifferentialRequest {
  symptoms: string[];
  candidates: {
    disease: string;
    fused_score: number;
    mining_score: number;
    retrieval_score: number;
  }[];
  alpha: number;
  mode: string;
}

export interface DifferentialResponse {
  summary: string;
  backend: string;
}

export interface InsightsResponse {
  alpha_sweep: { alpha: number; "recall@1": number; "recall@3": number;
                 "recall@5": number; "recall@10": number; mrr: number }[];
  ablation: Record<string, any>[];
  rules_per_disease: { disease: string; rules: number }[];
  source_distribution: { source_id: string; label: string; tier: number;
                          count: number }[];
  latency_recent: { ts: number; total_ms: number; encode_ms: number;
                     vector_search_ms: number; mining_ms: number;
                     explain_ms: number; backend: string;
                     explainer: string }[];
  headline_latency: { stage: string; n: number; mean_ms: number;
                      p50_ms: number; p95_ms: number; max_ms: number }[];
}

export const api = {
  async config(): Promise<ConfigDTO> {
    return getJson<ConfigDTO>(`${API_BASE}/config`);
  },
  async symptoms(): Promise<{ symptoms: string[]; count: number }> {
    return getJson(`${API_BASE}/symptoms`);
  },
  async sources(): Promise<{ sources: SourceDTO[] }> {
    return getJson(`${API_BASE}/sources`);
  },
  async diagnose(req: DiagnoseRequest): Promise<DiagnoseResponse> {
    const r = await fetch(`${API_BASE}/diagnose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`diagnose ${r.status}: ${text}`);
    }
    return r.json();
  },
  async suggest(symptoms: string[]): Promise<SuggestResponse> {
    const r = await fetch(`${API_BASE}/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms, top_n: 5 }),
    });
    if (!r.ok) throw new Error(`suggest ${r.status}`);
    return r.json();
  },
  async explainSymptom(symptom: string): Promise<ExplainSymptomResponse> {
    const r = await fetch(`${API_BASE}/explain_symptom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptom }),
    });
    if (!r.ok) throw new Error(`explain_symptom ${r.status}`);
    return r.json();
  },
  async differential(req: DifferentialRequest): Promise<DifferentialResponse> {
    const r = await fetch(`${API_BASE}/differential`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(`differential ${r.status}`);
    return r.json();
  },
  async insights(): Promise<InsightsResponse> {
    return getJson<InsightsResponse>(`${API_BASE}/insights`);
  },
  /**
   * Streamed diagnose. Calls onStage as each pipeline stage completes
   * server-side, then onComplete with the full DiagnoseResponse, then
   * resolves. If the server emits an error event we reject. AbortSignal
   * lets the caller cancel mid-flight (e.g., user navigated away).
   */
  async diagnoseStream(
    req: DiagnoseRequest,
    handlers: {
      onStage?: (s: PipelineStage) => void;
      onComplete?: (r: DiagnoseResponse) => void;
      onError?: (msg: string) => void;
    },
    signal?: AbortSignal
  ): Promise<void> {
    const r = await fetch(`${API_BASE}/diagnose/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
    if (!r.ok || !r.body) {
      const text = await r.text().catch(() => "");
      throw new Error(`diagnose/stream ${r.status}: ${text || "no body"}`);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    // Parse SSE: events are separated by '\n\n', each event has 'event:'
    // and 'data:' lines. Parse strict; reject on malformed JSON to avoid
    // silently swallowing data.
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = "message";
        let data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        try {
          const payload = JSON.parse(data);
          if (event === "stage") handlers.onStage?.(payload as PipelineStage);
          else if (event === "complete") handlers.onComplete?.(payload as DiagnoseResponse);
          else if (event === "error") handlers.onError?.(String(payload?.detail ?? ""));
        } catch (e: any) {
          throw new Error(`SSE parse failure on '${event}': ${e.message}`);
        }
      }
    }
  },
};
