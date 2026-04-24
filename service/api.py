"""FastAPI inference microservice.

Wraps the entire diagnostic pipeline behind a small REST surface:

    GET  /healthz                  -- liveness
    GET  /symptoms                 -- list of all symptom tokens
    GET  /sources                  -- list of distinct MedQuAD sources
                                       (used by the UI's source filter)
    POST /diagnose                 -- run the full pipeline for a query
    GET  /config                   -- which backends/explainers are available

The Next.js UI talks only to this service; the Python data tier and the
ML models stay behind it. CORS is open by default so the dev UI on
localhost:3000 can call us; tighten in prod.

Owner: Vineet wrote the schemas + endpoints, Sakshat the cache wiring,
Aishwarya the response shape (matches the Next.js card component).
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env from the project root before any module reads OPENAI_API_KEY,
# PINECONE_API_KEY, etc. We don't override existing env vars (so a CI/prod
# environment that exports them at the shell level still wins).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv is in requirements.txt; older envs survive without it.

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.clinical_explanation import ClinicalExplanation, get_clinical_explainer  # noqa: E402
from src.cross_encoder_rerank import CrossEncoderReranker  # noqa: E402
from src.pinecone_rerank import PineconeReranker  # noqa: E402
from src.disease_keywords import DISEASE_KEYWORDS  # noqa: E402
from src.evidence import cards_for_disease  # noqa: E402
from src.fusion_reranker import fuse  # noqa: E402
from src.mining_scorer import MiningScorer  # noqa: E402
from src.retrieval import DenseRetriever  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def _chat_with_token_fallback(client, *, model: str, messages: list,
                                max_token_budget: int, temperature: float = 0.2):
    """Call chat.completions while papering over the gpt-4o vs gpt-5 quirks.

    GPT-5-class deployments (e.g. Azure 'gpt-5.3-chat') diverge from the
    legacy chat-completions schema in two ways: they reject `max_tokens` in
    favour of `max_completion_tokens`, and they reject any non-default
    `temperature`. Older deployments (gpt-4o-mini, etc.) still take both.
    We try the modern shape first; on a 400 that names the offending param
    we retry with the legacy shape, dropping `temperature` if needed.
    """
    def _attempt(kwargs):
        return client.chat.completions.create(
            model=model, messages=messages, **kwargs,
        )

    new_kwargs = {"temperature": temperature, "max_completion_tokens": max_token_budget}
    try:
        return _attempt(new_kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)

    # Drop temperature if the model only allows the default.
    if "temperature" in msg:
        new_kwargs.pop("temperature", None)
        try:
            return _attempt(new_kwargs)
        except Exception as exc2:  # noqa: BLE001
            msg = str(exc2)

    # Fall back to the legacy `max_tokens` shape.
    if "max_tokens" in msg or "max_completion_tokens" in msg:
        legacy = {"max_tokens": max_token_budget}
        if "temperature" not in msg:
            legacy["temperature"] = temperature
        return _attempt(legacy)

    raise RuntimeError(f"chat.completions failed: {msg}")


# ---- Pydantic schemas -----------------------------------------------------

class DiagnoseRequest(BaseModel):
    symptoms: list[str] = Field(..., min_length=1, max_length=20)
    backend: str = Field("pubmedbert",
                          description="minilm | pubmedbert | azure-openai")
    mode: str = Field("fused", description="fused | mining-only | retrieval-only")
    alpha: float = Field(0.3, ge=0.0, le=1.0)
    expand_synonyms: bool = True
    cross_encoder: bool = False
    explainer: str = Field("auto", description="auto | template | openai")
    source_filter: Optional[str] = None
    passage_type_filter: Optional[str] = None
    top_n: int = Field(5, ge=1, le=15)
    # Configurable retrieval knobs. Defaults match the headline numbers in
    # the report; the UI exposes sliders for the demo.
    top_k_retrieval: int = Field(30, ge=5, le=80,
        description="How deep to search the vector store before disease attribution")
    related_top_k: int = Field(5, ge=1, le=15,
        description="How many 'nearby passages' to surface in the related-context rail")
    max_evidence_cards: int = Field(5, ge=1, le=10,
        description="Cap on evidence cards rendered per ranked diagnosis")
    # Optional Pinecone index override. When unset we fall back to whatever
    # PINECONE_INDEX_NAME points at. The two indexes we ship are
    # 255-data-mining (3072d, paired with azure-openai) and
    # medical-rag-medquad (768d, paired with pubmedbert).
    pinecone_index: Optional[str] = None
    trace: bool = Field(False, description="Include detailed pipeline trace in response")


class EvidenceClaimDTO(BaseModel):
    sentence: str
    matched_terms: list[str]
    char_start: int
    char_end: int


class EvidenceCardDTO(BaseModel):
    passage_id: str
    source_id: str
    source_label: str
    source_tier: int
    passage_type: str
    focus: str
    question: str
    full_text: str
    claims: list[EvidenceClaimDTO]
    specificity: float
    retrieval_score: float
    ce_score: Optional[float] = None


class MatchingRuleDTO(BaseModel):
    antecedent: list[str]
    confidence: float
    lift: float
    size: int


class DiagnosisDTO(BaseModel):
    disease: str
    disease_pretty: str
    fused_score: float
    mining_score: float
    retrieval_score: float
    matching_rules: list[MatchingRuleDTO]
    evidence_cards: list[EvidenceCardDTO]
    explanation: dict[str, Any]


class RelatedContextDTO(BaseModel):
    passage_id: str
    source: str
    source_label: str
    focus: str
    question: str
    text: str
    score: float


class PipelineStageDTO(BaseModel):
    """One observable stage of the diagnose pipeline.

    `data` carries whatever structured payload the stage produced or
    consumed (encoded query string, top matches, fired rules, fused
    table, prompt sent to LLM, etc.). The UI uses it to render an
    expandable inspector for each stage. Kept loosely typed because the
    payload shape varies by stage.
    """
    key: str               # stable id, e.g. "encode_query"
    label: str             # human-readable, e.g. "Encode query"
    detail: str            # one-liner caption shown in the timeline
    ms: float              # measured wall time
    data: dict[str, Any]   # structured payload (data flowing through the stage)


class DiagnoseResponse(BaseModel):
    query_symptoms: list[str]
    used_alpha: float
    used_backend: str
    used_mode: str
    explainer_backend: str
    vector_store: str
    diagnoses: list[DiagnosisDTO]
    related_context: list[RelatedContextDTO]
    latency_ms: dict[str, float]
    pipeline_trace: list[PipelineStageDTO] = []


# ---- Lifespan: load models once at boot -----------------------------------

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] booting...", flush=True)
    STATE["miner"] = MiningScorer.from_file(PROCESSED / "association_rules.csv")
    STATE["diseases"] = set(pd.read_csv(PROCESSED / "transactions.csv")["condition"].unique())
    df = pd.read_csv(PROCESSED / "transactions.csv").fillna("")
    syms: set[str] = set()
    for s in df["symptoms"]:
        syms.update(t for t in s.split("|") if t)
    STATE["symptoms"] = sorted(syms)
    STATE["retrievers"] = {}
    STATE["cross_encoder"] = None  # lazy
    print(f"[api] miner: {len(STATE['miner'].diseases)} diseases | "
          f"symptoms: {len(STATE['symptoms'])}", flush=True)
    yield
    print("[api] shutting down", flush=True)


# Known backend ↔ Pinecone index pairings. Any backend not listed here
# can still be used, but only if the user passes an explicit
# `pinecone_index` that we trust them to have set up.
_PINECONE_DEFAULT_INDEX = {
    "azure-openai": "255-data-mining",        # 3072d
    "pubmedbert":   "medical-rag-medquad",    # 768d
    # minilm has no Pinecone index by default; the user has to pass one
    # explicitly if they want to run minilm against Pinecone.
}


def _resolve_pinecone_index(backend: str, override: str | None) -> str | None:
    """Pick the index a request should target. `override` from the request
    wins; otherwise we use the curated mapping; otherwise None (caller
    will fall back to PINECONE_INDEX_NAME env)."""
    return override or _PINECONE_DEFAULT_INDEX.get(backend)


def _validate_pinecone_combo(backend: str, mode: str,
                                pinecone_index: str | None) -> None:
    """Raise 400 if the (backend, index) combo would mismatch dimensions.

    We trust the curated map; if the user passes an explicit index that
    isn't in the map we let the call through (their problem if they
    seeded it with the wrong dim). The check applies only when running
    against Pinecone with retrieval enabled.
    """
    if os.environ.get("VECTOR_STORE", "faiss").lower() != "pinecone":
        return
    if mode == "mining-only":
        return
    resolved = _resolve_pinecone_index(backend, pinecone_index)
    # If we have no index for this backend at all, that's a 400.
    if resolved is None:
        raise HTTPException(400,
            f"backend={backend} has no Pinecone index pre-configured. "
            f"Pass `pinecone_index` explicitly, or use one of: "
            f"{', '.join(_PINECONE_DEFAULT_INDEX.keys())}.")


def _get_retriever(backend: str,
                     index_name: str | None = None) -> DenseRetriever:
    """Return a cached retriever for (backend, index) pair.

    Each combination caches independently because the encoder model and
    Pinecone index are different. The default key (no index_name) keeps
    backward compatibility with anything that doesn't care about the
    index switch.
    """
    cache_key = f"{backend}|{index_name or ''}"
    if cache_key not in STATE["retrievers"]:
        STATE["retrievers"][cache_key] = DenseRetriever.from_files(
            backend, STATE["diseases"], index_name=index_name)
    return STATE["retrievers"][cache_key]


def _get_cross_encoder():
    """Lazy-load whichever reranker is configured.

    Order of preference:
        1. Pinecone Inference (Cohere Rerank 3.5) when running on
           Pinecone AND PINECONE_RERANK_MODEL is set.
        2. Local sentence-transformers cross-encoder (default fallback).
    """
    if STATE["cross_encoder"] is not None:
        return STATE["cross_encoder"]
    using_pinecone = os.environ.get("VECTOR_STORE", "faiss").lower() == "pinecone"
    pc_rerank_set = bool(os.environ.get("PINECONE_RERANK_MODEL"))
    if using_pinecone and pc_rerank_set:
        STATE["cross_encoder"] = PineconeReranker.load()
    else:
        STATE["cross_encoder"] = CrossEncoderReranker.load()
    return STATE["cross_encoder"]


# ---- App ------------------------------------------------------------------

app = FastAPI(title="Record-Based Medical Diagnostic Assistant",
                version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                     "http://localhost:3010", "http://127.0.0.1:3010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True, "diseases": len(STATE.get("diseases", []))}


@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": STATE["symptoms"], "count": len(STATE["symptoms"])}


@app.get("/sources")
def get_sources():
    """Distinct MedQuAD sources currently in the corpus."""
    # We compute this lazily off the in-memory passage list.
    retriever = _get_retriever("pubmedbert")
    counter = Counter(p["source"] for p in retriever.passages)
    from src.evidence import SOURCE_AUTHORITY, DEFAULT_SOURCE
    out = []
    for src, n in counter.most_common():
        meta = SOURCE_AUTHORITY.get(src, DEFAULT_SOURCE)
        out.append({"source_id": src, "label": meta["label"],
                     "tier": meta["tier"], "count": n})
    return {"sources": out}


@app.get("/config")
def get_config():
    """Surface which backends / explainers are available right now."""
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    vector_store = os.environ.get("VECTOR_STORE", "faiss")
    backends = ["minilm", "pubmedbert"]
    if has_openai:
        backends.append("azure-openai")
    return {
        "backends": backends,
        "explainers": (["template", "openai"] if has_openai else ["template"]),
        "vector_store": vector_store,
        "metadata_filter_supported": True,
        "openai_available": has_openai,
        "default_alpha": 0.3,
        "pinecone_index": os.environ.get("PINECONE_INDEX_NAME") if vector_store == "pinecone" else None,
        "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL") if has_openai else None,
        "rerank_model": os.environ.get("PINECONE_RERANK_MODEL") if vector_store == "pinecone" else None,
    }


# In-process ring buffer of recent /diagnose latency totals so the
# Insights dashboard can show a live histogram without persistence.
_LATENCY_HISTORY: list[dict[str, Any]] = []
_LATENCY_HISTORY_CAP = 50


@app.get("/insights")
def insights():
    """Aggregates that power the Insights dashboard.

    Reads the artefacts already produced by the evaluation harness
    (`data/results/*.csv` + `data/processed/association_rules.csv`) so the
    dashboard reflects the same numbers reported in the paper. No live
    computation here beyond the latency ring buffer.
    """
    results_dir = ROOT / "data" / "results"
    processed_dir = ROOT / "data" / "processed"

    out: dict[str, Any] = {
        "alpha_sweep": [],
        "ablation": [],
        "rules_per_disease": [],
        "source_distribution": [],
        "latency_recent": list(_LATENCY_HISTORY),
        "headline_latency": [],
    }

    # Alpha sweep curve.
    alpha_csv = results_dir / "alpha_sweep.csv"
    if alpha_csv.exists():
        try:
            df = pd.read_csv(alpha_csv)
            out["alpha_sweep"] = df.to_dict(orient="records")
        except Exception:  # noqa: BLE001
            pass

    # Ablation summary.
    abl_csv = results_dir / "ablation_summary.csv"
    if abl_csv.exists():
        try:
            df = pd.read_csv(abl_csv)
            out["ablation"] = df.to_dict(orient="records")
        except Exception:  # noqa: BLE001
            pass

    # Top-25 diseases by rule count.
    rules_csv = processed_dir / "association_rules.csv"
    if rules_csv.exists():
        try:
            df = pd.read_csv(rules_csv)
            counts = (df.groupby("consequent").size()
                       .sort_values(ascending=False).head(25))
            out["rules_per_disease"] = [
                {"disease": d.replace("_", " "), "rules": int(n)}
                for d, n in counts.items()
            ]
        except Exception:  # noqa: BLE001
            pass

    # MedQuAD source distribution (uses the in-memory passage list).
    try:
        retriever = _get_retriever("pubmedbert")
        from collections import Counter as _Counter
        from src.evidence import SOURCE_AUTHORITY, DEFAULT_SOURCE
        c = _Counter(p["source"] for p in retriever.passages)
        rows = []
        for src, n in c.most_common():
            meta = SOURCE_AUTHORITY.get(src, DEFAULT_SOURCE)
            rows.append({"source_id": src, "label": meta["label"],
                          "tier": int(meta["tier"]), "count": int(n)})
        out["source_distribution"] = rows
    except Exception:  # noqa: BLE001
        pass

    # Headline latency table (mean / p50 / p95 per stage from latency_summary.csv).
    lat_csv = results_dir / "latency_summary.csv"
    if lat_csv.exists():
        try:
            df = pd.read_csv(lat_csv)
            out["headline_latency"] = df.to_dict(orient="records")
        except Exception:  # noqa: BLE001
            pass

    return out


class SuggestRequest(BaseModel):
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    top_n: int = Field(5, ge=1, le=10)


@app.post("/suggest")
def suggest(req: SuggestRequest):
    """Suggest next symptoms most predictive given the current set.

    Uses the FP-Growth rule table: for each candidate symptom s, score it as
    the max confidence of any rule whose antecedent is (current ∪ {s}) and
    whose consequent is some disease. We exclude diseases-as-suggestions
    by construction (DX:* never appears in antecedent).

    Free, instant, no LLM. Falls back to "most common co-occurring symptoms"
    if the rule lookup is empty.
    """
    miner: MiningScorer = STATE["miner"]
    current = set(req.symptoms or [])
    scored: dict[str, tuple[float, int]] = {}  # token -> (best_conf, n_rules)

    for disease, ruleset in miner.rules_by_disease.items():
        for ante, conf, _lift in ruleset:
            # Want rules that contain at least one current symptom AND
            # at least one new candidate.
            extras = ante - current
            if not extras:
                continue
            overlap = len(ante & current) / max(1, len(ante))
            if overlap == 0:
                continue
            for tok in extras:
                cur = scored.get(tok, (0.0, 0))
                score = conf * overlap
                if score > cur[0]:
                    scored[tok] = (score, cur[1] + 1)
                else:
                    scored[tok] = (cur[0], cur[1] + 1)

    if not scored and current:
        # Cold-start: rank tokens by global co-occurrence with current symptoms.
        df = pd.read_csv(PROCESSED / "transactions.csv").fillna("")
        co: Counter = Counter()
        for syms_str in df["symptoms"]:
            syms = set(t for t in syms_str.split("|") if t)
            if syms & current:
                co.update(s for s in syms if s not in current)
        return {
            "suggestions": [
                {"symptom": s, "score": float(n) / sum(co.values()),
                 "n_rules": 0}
                for s, n in co.most_common(req.top_n)
            ]
        }

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    return {
        "suggestions": [
            {"symptom": tok, "score": s, "n_rules": n}
            for tok, (s, n) in ranked[:req.top_n]
        ]
    }


# Cache for /explain_symptom so a clinician clicking the same chip multiple
# times doesn't re-bill OpenAI.
_SYMPTOM_EXPLAIN_CACHE: dict[str, dict[str, Any]] = {}


class ExplainSymptomRequest(BaseModel):
    symptom: str = Field(..., min_length=1, max_length=80)


@app.post("/explain_symptom")
def explain_symptom(req: ExplainSymptomRequest):
    """Return a 1-2 sentence clinical gloss for a single symptom token.

    Uses GPT-4o-mini when OPENAI_API_KEY is set; falls back to a curated
    short description from the synonym dictionary otherwise. Cached
    in-process by symptom token so repeated clicks are free.
    """
    tok = req.symptom.strip().lower()
    if not tok:
        raise HTTPException(400, "empty symptom")
    if tok in _SYMPTOM_EXPLAIN_CACHE:
        return _SYMPTOM_EXPLAIN_CACHE[tok]

    from src.synonym_expansion import SYMPTOM_SYNONYMS
    syns = SYMPTOM_SYNONYMS.get(tok, [])
    pretty = tok.replace("_", " ")

    # Pull diseases this symptom is most predictive of from the FP-Growth
    # rules, so even the template fallback can say something specific.
    miner: MiningScorer = STATE["miner"]
    disease_hits: list[tuple[str, float]] = []
    for disease, ruleset in miner.rules_by_disease.items():
        best = 0.0
        for ante, conf, _lift in ruleset:
            if tok in ante:
                if conf > best:
                    best = conf
        if best > 0:
            disease_hits.append((disease, best))
    disease_hits.sort(key=lambda kv: -kv[1])
    top_dx = [d.replace("_", " ") for d, _ in disease_hits[:3]]

    def _template_text() -> str:
        bits = [f"\"{pretty}\""]
        if syns:
            bits.append(f"clinically: {', '.join(syns)}")
        if top_dx:
            bits.append(f"appears in mined rules for {', '.join(top_dx)}")
        return ". ".join(bits) + "."

    if not os.environ.get("OPENAI_API_KEY"):
        out = {"symptom": tok, "pretty": pretty, "synonyms": syns,
               "explanation": _template_text(), "backend": "template",
               "top_diseases": top_dx}
        _SYMPTOM_EXPLAIN_CACHE[tok] = out
        return out

    try:
        from openai import OpenAI
        client = OpenAI()
        sys_prompt = (
            "You are a clinical reference assistant. In 2 short sentences, "
            "explain what a symptom token typically means in clinical "
            "practice. Be concrete: mention common contexts (e.g. "
            "infections, cardiovascular, etc.) and one or two formal "
            "clinical synonyms if applicable. No disclaimers."
        )
        user = (f"Symptom token: {pretty}"
                + (f"\nKnown clinical synonyms: {', '.join(syns)}" if syns else ""))
        resp = _chat_with_token_fallback(
            client,
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            # 180 was right for gpt-4o-mini; reasoning models like gpt-5
            # burn most of the budget on internal thought before emitting
            # visible output, so we budget more generously.
            max_token_budget=900,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        out = {"symptom": tok, "pretty": pretty, "synonyms": syns,
               "explanation": _template_text(),
               "backend": "openai-failed:template",
               "top_diseases": top_dx}
        _SYMPTOM_EXPLAIN_CACHE[tok] = out
        return out

    out = {"symptom": tok, "pretty": pretty, "synonyms": syns,
           "explanation": text, "backend": "openai:gpt-4o-mini",
           "top_diseases": top_dx}
    _SYMPTOM_EXPLAIN_CACHE[tok] = out
    return out


class DifferentialCandidate(BaseModel):
    disease: str
    fused_score: float
    mining_score: float
    retrieval_score: float


class DifferentialRequest(BaseModel):
    symptoms: list[str] = Field(..., min_length=1, max_length=20)
    candidates: list[DifferentialCandidate] = Field(..., min_length=1, max_length=5)
    alpha: float = 0.3
    mode: str = "fused"


# Cache: (sorted symptoms, sorted top-3 disease names) -> response.
_DIFFERENTIAL_CACHE: dict[tuple, dict[str, Any]] = {}


@app.post("/differential")
def differential(req: DifferentialRequest):
    """One-paragraph differential-diagnosis summary in clinician-note voice.

    Renders something like:
        "55-year-old presenting with chest pain, sweating, lightheadedness,
         and breathlessness. Mining and retrieval converge on heart_attack
         (fused 0.83). Two competing entities -- diabetes (0.16) and
         bronchial_asthma (0.12) -- are surfaced primarily by retrieval and
         lack matching mined rules. Recommend: ECG, troponin."

    Cached by (symptom set × top-3 diseases) tuple so repeated diagnoses
    of the same query don't re-bill OpenAI. Falls back gracefully to a
    deterministic template stitch if the LLM call fails (mirrors how the
    per-diagnosis explainer behaves).
    """
    syms_pretty = ", ".join(s.replace("_", " ") for s in req.symptoms)
    top3 = req.candidates[:3]
    cache_key = (
        tuple(sorted(req.symptoms)),
        tuple(sorted(c.disease for c in top3)),
        round(req.alpha, 2),
        req.mode,
    )
    if cache_key in _DIFFERENTIAL_CACHE:
        return _DIFFERENTIAL_CACHE[cache_key]

    def _template_text() -> str:
        bits: list[str] = []
        bits.append(
            f"Patient presents with {syms_pretty}."
        )
        if top3:
            top = top3[0]
            bits.append(
                f"The hybrid pipeline ranks {top.disease.replace('_', ' ')} "
                f"first (fused {top.fused_score:.2f}; mining "
                f"{top.mining_score:.2f}, retrieval {top.retrieval_score:.2f})."
            )
        if len(top3) > 1:
            others = ", ".join(
                f"{c.disease.replace('_', ' ')} ({c.fused_score:.2f})"
                for c in top3[1:]
            )
            bits.append(
                f"Competing entities considered: {others}."
            )
        bits.append(
            "This is a data-driven prior; physical exam, labs, and imaging "
            "remain required for definitive diagnosis."
        )
        return " ".join(bits)

    if not os.environ.get("OPENAI_API_KEY"):
        out = {
            "summary": _template_text(),
            "backend": "template",
        }
        _DIFFERENTIAL_CACHE[cache_key] = out
        return out

    rule_block = "\n".join(
        f"  {i+1}. {c.disease.replace('_', ' ')} — fused {c.fused_score:.3f} "
        f"(mining {c.mining_score:.3f}, retrieval {c.retrieval_score:.3f})"
        for i, c in enumerate(top3)
    )
    user = (
        f"Patient symptoms: {syms_pretty}\n"
        f"Pipeline mode: {req.mode}, fusion weight α={req.alpha:.2f}\n"
        f"Top-3 ranked candidates:\n{rule_block}\n\n"
        "Write a 3-4 sentence differential-diagnosis summary in the voice "
        "of a brief clinician note. Lead with the patient's presentation. "
        "Name the top candidate and the numerical evidence. Acknowledge "
        "the competing candidates and what the system cannot determine. "
        "End with one or two concrete next steps a clinician would take "
        "(e.g., ECG, troponin, CBC, imaging) appropriate to the top "
        "candidate. Tone: concise, clinical, hedged. No disclaimers, "
        "no bullet points, no headings — one paragraph of prose."
    )
    sys_prompt = (
        "You are a clinical decision-support assistant generating concise "
        "differential-diagnosis summaries. You produce prose, not lists."
    )
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = _chat_with_token_fallback(
            client,
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            max_token_budget=1200,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        out = {
            "summary": text,
            "backend": f"openai:{os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}",
        }
    except Exception:  # noqa: BLE001
        out = {
            "summary": _template_text(),
            "backend": "openai-failed:template",
        }

    _DIFFERENTIAL_CACHE[cache_key] = out
    return out


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    """Full pipeline. When `trace=True`, instrument every observable stage
    (12 in the fused path) and return both the timings and the data each
    stage produced/consumed. The trace is what the UI's expandable timeline
    inspector renders.

    The non-traced path is unchanged in behavior; tracing adds at most a
    few dict copies and a couple of small numpy norm computations, well
    under 1ms total overhead.
    """
    if not req.symptoms:
        raise HTTPException(400, "no symptoms supplied")
    if req.mode not in ("fused", "mining-only", "retrieval-only"):
        raise HTTPException(400, f"bad mode: {req.mode}")
    if req.backend not in ("minilm", "pubmedbert", "azure-openai"):
        raise HTTPException(400, f"bad backend: {req.backend}")
    _validate_pinecone_combo(req.backend, req.mode, req.pinecone_index)
    resolved_index = _resolve_pinecone_index(req.backend, req.pinecone_index)

    from src.synonym_expansion import (SYMPTOM_SYNONYMS,
                                            expand_query_string)
    from src.retrieval import _candidate_diseases_from_passage  # noqa: SLF001
    from src.evidence import SOURCE_AUTHORITY, DEFAULT_SOURCE

    miner: MiningScorer = STATE["miner"]
    timings: dict[str, float] = {}
    trace: list[dict[str, Any]] = []

    def _stage(key: str, label: str, detail: str, ms: float, data: dict):
        # Cap any list payload to keep the response from ballooning when a
        # caller asks for trace=True. The caller can request more later if
        # we add a paged trace endpoint.
        timings[f"{key}_ms"] = round(ms, 2)
        if req.trace:
            trace.append({"key": key, "label": label, "detail": detail,
                            "ms": round(ms, 2), "data": data})

    metadata_filter: dict[str, Any] | None = None
    if req.source_filter:
        metadata_filter = {"source": req.source_filter}

    t0 = time.perf_counter()

    # ---------------- Retrieval path (skipped in mining-only) ------------
    retr_scores: dict[str, float] = {}
    retr_passages: dict[str, list[dict]] = {}
    related: list[dict] = []
    raw_matches_for_trace: list[dict] = []
    query_for_trace: str = ""

    if req.mode != "mining-only":
        retr = _get_retriever(req.backend, resolved_index)

        # Stage 1: build the query string from symptoms.
        t = time.perf_counter()
        base_query = "symptoms: " + ", ".join(
            s.replace("_", " ") for s in req.symptoms)
        ms = (time.perf_counter() - t) * 1000.0
        _stage("build_query", "Build query string",
                f"{len(req.symptoms)} symptoms → natural-language probe",
                ms, {"query": base_query})

        # Stage 2: optional synonym expansion.
        t = time.perf_counter()
        if req.expand_synonyms:
            query = expand_query_string(base_query, req.symptoms)
            applied = {s: SYMPTOM_SYNONYMS.get(s, []) for s in req.symptoms
                        if SYMPTOM_SYNONYMS.get(s)}
        else:
            query = base_query
            applied = {}
        query_for_trace = query
        ms = (time.perf_counter() - t) * 1000.0
        _stage("expand_synonyms", "Synonym expansion",
                f"{sum(len(v) for v in applied.values())} clinical terms appended"
                if applied else "skipped (toggle off)",
                ms,
                {"applied": applied, "expanded_query": query,
                 "delta_chars": len(query) - len(base_query)})

        # Stage 3: encode the query into a dense vector.
        t = time.perf_counter()
        q_vec = retr.backend.encode([query])
        ms = (time.perf_counter() - t) * 1000.0
        norm = float(np.linalg.norm(q_vec[0])) if q_vec.shape[0] else 0.0
        _stage("encode", "Encode query",
                f"{retr.backend.name} → {retr.backend.dim}d vector",
                ms,
                {"backend": retr.backend.name, "dim": int(retr.backend.dim),
                 "vector_norm": round(norm, 4),
                 "vector_preview": [round(float(x), 4)
                                      for x in q_vec[0][:8].tolist()]})

        # Stage 4: vector store search.
        t = time.perf_counter()
        matches = retr.store.query(q_vec, top_k=req.top_k_retrieval,
                                      filter=metadata_filter)
        ms = (time.perf_counter() - t) * 1000.0
        store_kind = ("pinecone"
                       if os.environ.get("VECTOR_STORE",
                                            "faiss").lower() == "pinecone"
                       else "faiss")
        # Capture the top 10 raw matches before disease attribution; the
        # full 30 are used for ranking but 10 is the right size for an
        # inspector panel.
        raw_matches_for_trace = [
            {"passage_id": m.passage_id,
             "score": round(float(m.score), 4),
             "source": (m.metadata or {}).get("source", ""),
             "focus": (m.metadata or {}).get("focus", ""),
             "question": (m.metadata or {}).get("question", "")[:120]}
            for m in matches[:10]
        ]
        _stage("vector_search", "Vector search",
                f"{store_kind} · top-{len(matches)} cosine matches"
                + (f" · filter={metadata_filter}" if metadata_filter else ""),
                ms,
                {"store": store_kind, "top_k": len(matches),
                 "filter": metadata_filter, "top_matches": raw_matches_for_trace})

        # Stage 5: disease attribution. Map matches to Kaggle disease universe.
        t = time.perf_counter()
        per_disease_score: dict[str, float] = {}
        per_disease_passages: dict[str, list[dict]] = {}
        attribution_breakdown: list[dict] = []
        for m in matches:
            md = m.metadata or {}
            p = {"id": m.passage_id, "source": md.get("source", ""),
                 "focus": md.get("focus", ""),
                 "question": md.get("question", ""),
                 "text": md.get("text", "")}
            cands = _candidate_diseases_from_passage(p, retr.disease_universe)
            if not cands:
                continue
            for d in cands:
                if m.score > per_disease_score.get(d, -1.0):
                    per_disease_score[d] = float(m.score)
                per_disease_passages.setdefault(d, []).append(
                    {**p, "score": float(m.score)})
            if len(attribution_breakdown) < 8:
                attribution_breakdown.append(
                    {"passage_id": p["id"],
                     "focus": p["focus"],
                     "matched_diseases": cands,
                     "score": round(float(m.score), 4)})
        retr_scores = per_disease_score
        retr_passages = per_disease_passages
        ms = (time.perf_counter() - t) * 1000.0
        _stage("attribute", "Disease attribution",
                f"{len(matches)} passages → {len(per_disease_score)} of "
                f"{len(retr.disease_universe)} disease classes",
                ms,
                {"diseases_hit": len(per_disease_score),
                 "universe_size": len(retr.disease_universe),
                 "sample": attribution_breakdown,
                 "per_disease_top_score":
                     dict(sorted(per_disease_score.items(),
                                  key=lambda kv: -kv[1])[:8])})

        # Stage 6: related-context (separate top-K query, no disease filter).
        t = time.perf_counter()
        related_matches = retr.store.query(q_vec, top_k=req.related_top_k,
                                              filter=metadata_filter)
        related = []
        for m in related_matches:
            md = m.metadata or {}
            related.append(
                {"id": m.passage_id, "source": md.get("source", ""),
                 "focus": md.get("focus", ""),
                 "question": md.get("question", ""),
                 "text": md.get("text", ""),
                 "score": float(m.score)})
        ms = (time.perf_counter() - t) * 1000.0
        _stage("related", "Related context",
                f"top-{len(related)} nearby passages",
                ms,
                {"items": [{"focus": r["focus"], "source": r["source"],
                              "score": round(r["score"], 4)}
                             for r in related]})

    used_alpha = (0.0 if req.mode == "mining-only"
                   else 1.0 if req.mode == "retrieval-only"
                   else req.alpha)

    # ---------------- Mining ---------------------------------------------
    t = time.perf_counter()
    mine_scores: dict[str, float] = ({} if req.mode == "retrieval-only"
                                       else miner.score(req.symptoms))
    ms = (time.perf_counter() - t) * 1000.0
    # Surface up to 8 fired rules with overlap fraction for inspector.
    fired_for_trace: list[dict] = []
    if mine_scores:
        sym_set = set(req.symptoms)
        for disease, score in sorted(mine_scores.items(),
                                       key=lambda kv: -kv[1])[:8]:
            for ante, conf, lift in miner.rules_by_disease.get(disease, []):
                if ante.issubset(sym_set):
                    fired_for_trace.append(
                        {"disease": disease,
                         "antecedent": sorted(ante),
                         "confidence": round(conf, 3),
                         "lift": round(lift, 2),
                         "overlap": round(len(ante & sym_set) / max(1, len(ante)), 3),
                         "score": round(conf * (len(ante & sym_set) / max(1, len(ante))), 3)})
                    break  # one rule per disease for the inspector
    _stage("mining", "Mining scorer",
            f"FP-Growth · {len(mine_scores)} of {len(miner.diseases)} diseases scored",
            ms,
            {"diseases_scored": len(mine_scores),
             "rules_in_table": sum(len(v) for v in miner.rules_by_disease.values()),
             "top_fired": fired_for_trace})

    # ---------------- Fusion ---------------------------------------------
    t = time.perf_counter()
    candidates = fuse(retr_scores, mine_scores, alpha=used_alpha)[:req.top_n]
    ms = (time.perf_counter() - t) * 1000.0
    _stage("fuse", "Hybrid fusion",
            f"FusedScore = α·retrieval + (1-α)·mining · α={used_alpha:.2f}",
            ms,
            {"alpha": used_alpha, "mode": req.mode,
             "candidates": [
                 {"disease": c.disease,
                  "fused_score": round(c.fused_score, 4),
                  "mining_score": round(c.mining_score, 4),
                  "retrieval_score": round(c.retrieval_score, 4)}
                 for c in candidates]})

    # ---------------- Optional cross-encoder rerank ----------------------
    if req.cross_encoder and candidates and retr_passages:
        t = time.perf_counter()
        ce = _get_cross_encoder()
        q = "symptoms: " + ", ".join(s.replace("_", " ") for s in req.symptoms)
        ce_summary: list[dict] = []
        for cand in candidates[:3]:
            ps = retr_passages.get(cand.disease, [])
            if ps:
                before_top = ps[0]["id"] if ps else None
                retr_passages[cand.disease] = ce.rerank(q, ps[:15], top_k=10)
                after = retr_passages[cand.disease]
                ce_summary.append({
                    "disease": cand.disease,
                    "k": len(after),
                    "top_before": before_top,
                    "top_after": after[0]["id"] if after else None,
                })
        ms = (time.perf_counter() - t) * 1000.0
        _stage("cross_encoder", "Cross-encoder rerank",
                f"reranked top-{len(ce_summary)} candidates",
                ms, {"reranker": getattr(ce, "name", "cross-encoder"),
                       "candidates": ce_summary})

    # ---------------- Evidence + Explanation ----------------------------
    explainer = get_clinical_explainer(req.explainer)
    diagnoses_out: list[DiagnosisDTO] = []
    explain_total = 0.0
    evidence_summary: list[dict] = []
    explain_summary: list[dict] = []

    for cand in candidates:
        rules = miner.matching_rules(req.symptoms, cand.disease, top_n=3)
        cards = cards_for_disease(retr_passages.get(cand.disease, []),
                                     query_symptoms=req.symptoms,
                                     disease_keywords=DISEASE_KEYWORDS.get(cand.disease, []),
                                     max_cards=req.max_evidence_cards)
        if req.passage_type_filter:
            cards = [c for c in cards if c.passage_type == req.passage_type_filter]
        evidence_summary.append({
            "disease": cand.disease,
            "n_cards": len(cards),
            "top_card": (
                {"source": cards[0].source_label,
                 "tier": cards[0].source_tier,
                 "passage_type": cards[0].passage_type,
                 "specificity": round(cards[0].specificity, 3)}
                if cards else None),
        })

        t = time.perf_counter()
        try:
            expl = explainer.explain(disease=cand.disease,
                                       query_symptoms=req.symptoms,
                                       matching_rules=rules,
                                       evidence_cards=cards)
        except Exception as exc:  # noqa: BLE001
            from src.clinical_explanation import TemplateClinicalExplainer
            expl = TemplateClinicalExplainer().explain(
                disease=cand.disease, query_symptoms=req.symptoms,
                matching_rules=rules, evidence_cards=cards)
            expl.backend = f"fallback:{type(exc).__name__}"
        explain_total += (time.perf_counter() - t) * 1000.0
        explain_summary.append({
            "disease": cand.disease,
            "backend": expl.backend,
            "preview": expl.symptom_disease_link[:140] + (
                "…" if len(expl.symptom_disease_link) > 140 else ""),
            "citation_count": len(expl.citations),
        })

        diagnoses_out.append(DiagnosisDTO(
            disease=cand.disease,
            disease_pretty=cand.disease.replace("_", " ").title(),
            fused_score=cand.fused_score,
            mining_score=cand.mining_score,
            retrieval_score=cand.retrieval_score,
            matching_rules=[MatchingRuleDTO(**r) for r in rules],
            evidence_cards=[EvidenceCardDTO(**c.to_dict()) for c in cards],
            explanation=expl.to_dict(),
        ))

    # Two synthetic stages so the UI shows them as distinct rows; each one
    # bundles the per-candidate work above. We attribute the per-card cost
    # to "evidence" and the LLM cost to "explain".
    # (The explain timer above was started for each candidate; sum lives in
    # explain_total. Evidence-card construction is cheap and counted as the
    # remainder.)
    _stage("evidence", "Evidence cards",
            f"{len(candidates)} diagnoses · claim-level extraction · tier sort",
            0.0,  # accurate per-call timing is below the noise floor
            {"per_diagnosis": evidence_summary})
    _stage("explain", "Clinical explanation",
            f"{explainer.name} · 4-section JSON · {len(candidates)} diagnoses",
            explain_total,
            {"backend": explainer.name,
             "model": getattr(explainer, "model", None),
             "per_diagnosis": explain_summary})

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # Push to the ring buffer the Insights dashboard reads from. We keep the
    # per-stage breakdown so the dashboard can show stage-decomposed bars,
    # not just totals. Bounded to the last N requests.
    _LATENCY_HISTORY.append({
        "ts": time.time(),
        "total_ms": timings["total_ms"],
        "encode_ms": timings.get("encode_ms", 0.0),
        "vector_search_ms": timings.get("vector_search_ms", 0.0),
        "mining_ms": timings.get("mining_ms", 0.0),
        "explain_ms": timings.get("explain_ms", 0.0),
        "backend": req.backend,
        "explainer": req.explainer,
    })
    if len(_LATENCY_HISTORY) > _LATENCY_HISTORY_CAP:
        del _LATENCY_HISTORY[: len(_LATENCY_HISTORY) - _LATENCY_HISTORY_CAP]

    # ---------------- Build response DTO ---------------------------------
    related_dto = []
    for p in related[:5]:
        meta = SOURCE_AUTHORITY.get(p.get("source", ""), DEFAULT_SOURCE)
        related_dto.append(RelatedContextDTO(
            passage_id=str(p["id"]),
            source=p.get("source", ""),
            source_label=meta["label"],
            focus=p.get("focus", ""),
            question=p.get("question", ""),
            text=p.get("text", ""),
            score=float(p.get("score", 0.0)),
        ))

    return DiagnoseResponse(
        query_symptoms=req.symptoms,
        used_alpha=used_alpha,
        used_backend=("none" if req.mode == "mining-only" else req.backend),
        used_mode=req.mode,
        explainer_backend=explainer.name,
        vector_store=os.environ.get("VECTOR_STORE", "faiss"),
        diagnoses=diagnoses_out,
        related_context=related_dto,
        latency_ms=timings,
        pipeline_trace=[PipelineStageDTO(**s) for s in trace],
    )


# --------------------------------------------------------------------------
# Streaming variant: same pipeline, but each stage gets emitted as a
# Server-Sent Event the moment it completes server-side. The frontend's
# walkthrough engine consumes these and animates each stage as it arrives.
# --------------------------------------------------------------------------

@app.post("/diagnose/stream")
def diagnose_stream(req: DiagnoseRequest):
    """SSE-streamed pipeline. Emits 'stage' events as each phase finishes,
    plus a final 'complete' event with the full DiagnoseResponse payload.

    Event protocol (each event is one SSE message):
        event: stage
        data: {"key", "label", "detail", "ms", "data"}

        event: complete
        data: <full DiagnoseResponse JSON>

        event: error
        data: {"detail": "..."}

    The synchronous /diagnose endpoint stays in place; this one trades the
    single response for incremental events so the UI can react per stage.
    Validation errors return 400 immediately (not streamed) since there's
    nothing useful to stream when the request itself is malformed.
    """
    # Validation — same shape as /diagnose. Done up front so a 400
    # short-circuits before we open the stream.
    if not req.symptoms:
        raise HTTPException(400, "no symptoms supplied")
    if req.mode not in ("fused", "mining-only", "retrieval-only"):
        raise HTTPException(400, f"bad mode: {req.mode}")
    if req.backend not in ("minilm", "pubmedbert", "azure-openai"):
        raise HTTPException(400, f"bad backend: {req.backend}")
    _validate_pinecone_combo(req.backend, req.mode, req.pinecone_index)
    resolved_index = _resolve_pinecone_index(req.backend, req.pinecone_index)

    def event_stream():
        """Yield SSE-formatted bytes as each pipeline stage completes."""
        import json as _json
        from src.synonym_expansion import (SYMPTOM_SYNONYMS,
                                                expand_query_string)
        from src.retrieval import _candidate_diseases_from_passage  # noqa: SLF001
        from src.evidence import SOURCE_AUTHORITY, DEFAULT_SOURCE

        miner: MiningScorer = STATE["miner"]
        timings: dict[str, float] = {}
        trace: list[dict[str, Any]] = []

        def emit(event: str, payload: dict) -> bytes:
            # SSE wire format: 'event: NAME\ndata: JSON\n\n'.
            return (f"event: {event}\n"
                     f"data: {_json.dumps(payload, default=str)}\n\n").encode()

        def stage(key: str, label: str, detail: str, ms: float, data: dict):
            timings[f"{key}_ms"] = round(ms, 2)
            entry = {"key": key, "label": label, "detail": detail,
                       "ms": round(ms, 2), "data": data}
            trace.append(entry)
            return emit("stage", entry)

        try:
            metadata_filter: dict[str, Any] | None = None
            if req.source_filter:
                metadata_filter = {"source": req.source_filter}

            t0 = time.perf_counter()

            # ---------------- Retrieval path ----------------
            retr_scores: dict[str, float] = {}
            retr_passages: dict[str, list[dict]] = {}
            related: list[dict] = []

            if req.mode != "mining-only":
                retr = _get_retriever(req.backend, resolved_index)

                t = time.perf_counter()
                base_query = "symptoms: " + ", ".join(
                    s.replace("_", " ") for s in req.symptoms)
                ms = (time.perf_counter() - t) * 1000.0
                yield stage("build_query", "Build query string",
                              f"{len(req.symptoms)} symptoms → natural-language probe",
                              ms, {"query": base_query})

                t = time.perf_counter()
                if req.expand_synonyms:
                    query = expand_query_string(base_query, req.symptoms)
                    applied = {s: SYMPTOM_SYNONYMS.get(s, []) for s in req.symptoms
                                if SYMPTOM_SYNONYMS.get(s)}
                else:
                    query = base_query
                    applied = {}
                ms = (time.perf_counter() - t) * 1000.0
                yield stage("expand_synonyms", "Synonym expansion",
                              f"{sum(len(v) for v in applied.values())} clinical terms appended"
                              if applied else "skipped (toggle off)",
                              ms,
                              {"applied": applied, "expanded_query": query,
                               "delta_chars": len(query) - len(base_query)})

                t = time.perf_counter()
                q_vec = retr.backend.encode([query])
                ms = (time.perf_counter() - t) * 1000.0
                norm = float(np.linalg.norm(q_vec[0])) if q_vec.shape[0] else 0.0
                yield stage("encode", "Encode query",
                              f"{retr.backend.name} → {retr.backend.dim}d vector",
                              ms,
                              {"backend": retr.backend.name,
                               "dim": int(retr.backend.dim),
                               "vector_norm": round(norm, 4),
                               "vector_preview": [round(float(x), 4)
                                                    for x in q_vec[0][:8].tolist()]})

                t = time.perf_counter()
                matches = retr.store.query(q_vec, top_k=req.top_k_retrieval,
                                              filter=metadata_filter)
                ms = (time.perf_counter() - t) * 1000.0
                store_kind = ("pinecone"
                                if os.environ.get("VECTOR_STORE", "faiss").lower() == "pinecone"
                                else "faiss")
                top_matches = [
                    {"passage_id": m.passage_id,
                     "score": round(float(m.score), 4),
                     "source": (m.metadata or {}).get("source", ""),
                     "focus": (m.metadata or {}).get("focus", ""),
                     "question": (m.metadata or {}).get("question", "")[:120]}
                    for m in matches[:10]]
                yield stage("vector_search", "Vector search",
                              f"{store_kind} · top-{len(matches)} cosine matches"
                              + (f" · filter={metadata_filter}" if metadata_filter else ""),
                              ms,
                              {"store": store_kind, "top_k": len(matches),
                               "filter": metadata_filter, "top_matches": top_matches})

                t = time.perf_counter()
                per_disease_score: dict[str, float] = {}
                per_disease_passages: dict[str, list[dict]] = {}
                attribution_breakdown: list[dict] = []
                for m in matches:
                    md = m.metadata or {}
                    p = {"id": m.passage_id, "source": md.get("source", ""),
                         "focus": md.get("focus", ""),
                         "question": md.get("question", ""),
                         "text": md.get("text", "")}
                    cands = _candidate_diseases_from_passage(p, retr.disease_universe)
                    if not cands:
                        continue
                    for d in cands:
                        if m.score > per_disease_score.get(d, -1.0):
                            per_disease_score[d] = float(m.score)
                        per_disease_passages.setdefault(d, []).append(
                            {**p, "score": float(m.score)})
                    if len(attribution_breakdown) < 8:
                        attribution_breakdown.append(
                            {"passage_id": p["id"], "focus": p["focus"],
                             "matched_diseases": cands,
                             "score": round(float(m.score), 4)})
                retr_scores = per_disease_score
                retr_passages = per_disease_passages
                ms = (time.perf_counter() - t) * 1000.0
                yield stage("attribute", "Disease attribution",
                              f"{len(matches)} passages → {len(per_disease_score)} of "
                              f"{len(retr.disease_universe)} disease classes",
                              ms,
                              {"diseases_hit": len(per_disease_score),
                               "universe_size": len(retr.disease_universe),
                               "sample": attribution_breakdown,
                               "per_disease_top_score":
                                   dict(sorted(per_disease_score.items(),
                                                key=lambda kv: -kv[1])[:8])})

                t = time.perf_counter()
                related_matches = retr.store.query(q_vec, top_k=req.related_top_k,
                                                       filter=metadata_filter)
                related = []
                for m in related_matches:
                    md = m.metadata or {}
                    related.append(
                        {"id": m.passage_id, "source": md.get("source", ""),
                         "focus": md.get("focus", ""),
                         "question": md.get("question", ""),
                         "text": md.get("text", ""),
                         "score": float(m.score)})
                ms = (time.perf_counter() - t) * 1000.0
                yield stage("related", "Related context",
                              f"top-{len(related)} nearby passages", ms,
                              {"items": [{"focus": r["focus"], "source": r["source"],
                                            "score": round(r["score"], 4)}
                                           for r in related]})

            used_alpha = (0.0 if req.mode == "mining-only"
                            else 1.0 if req.mode == "retrieval-only"
                            else req.alpha)

            # ---------------- Mining ----------------
            t = time.perf_counter()
            mine_scores: dict[str, float] = ({} if req.mode == "retrieval-only"
                                                 else miner.score(req.symptoms))
            ms = (time.perf_counter() - t) * 1000.0
            fired_for_trace: list[dict] = []
            if mine_scores:
                sym_set = set(req.symptoms)
                for disease, score in sorted(mine_scores.items(),
                                                key=lambda kv: -kv[1])[:8]:
                    for ante, conf, lift in miner.rules_by_disease.get(disease, []):
                        if ante.issubset(sym_set):
                            fired_for_trace.append(
                                {"disease": disease,
                                 "antecedent": sorted(ante),
                                 "confidence": round(conf, 3),
                                 "lift": round(lift, 2),
                                 "overlap": round(len(ante & sym_set) / max(1, len(ante)), 3),
                                 "score": round(conf * (len(ante & sym_set) / max(1, len(ante))), 3)})
                            break
            yield stage("mining", "Mining scorer",
                          f"FP-Growth · {len(mine_scores)} of {len(miner.diseases)} diseases scored",
                          ms,
                          {"diseases_scored": len(mine_scores),
                           "rules_in_table": sum(len(v) for v in miner.rules_by_disease.values()),
                           "top_fired": fired_for_trace})

            # ---------------- Fusion ----------------
            t = time.perf_counter()
            candidates = fuse(retr_scores, mine_scores, alpha=used_alpha)[:req.top_n]
            ms = (time.perf_counter() - t) * 1000.0
            yield stage("fuse", "Hybrid fusion",
                          f"FusedScore = α·retrieval + (1-α)·mining · α={used_alpha:.2f}",
                          ms,
                          {"alpha": used_alpha, "mode": req.mode,
                           "candidates": [
                               {"disease": c.disease,
                                "fused_score": round(c.fused_score, 4),
                                "mining_score": round(c.mining_score, 4),
                                "retrieval_score": round(c.retrieval_score, 4)}
                               for c in candidates]})

            # ---------------- Optional cross-encoder ----------------
            if req.cross_encoder and candidates and retr_passages:
                t = time.perf_counter()
                ce = _get_cross_encoder()
                q = "symptoms: " + ", ".join(s.replace("_", " ") for s in req.symptoms)
                ce_summary: list[dict] = []
                for cand in candidates[:3]:
                    ps = retr_passages.get(cand.disease, [])
                    if ps:
                        before_top = ps[0]["id"] if ps else None
                        retr_passages[cand.disease] = ce.rerank(q, ps[:15], top_k=10)
                        after = retr_passages[cand.disease]
                        ce_summary.append({
                            "disease": cand.disease, "k": len(after),
                            "top_before": before_top,
                            "top_after": after[0]["id"] if after else None})
                ms = (time.perf_counter() - t) * 1000.0
                yield stage("cross_encoder", "Cross-encoder rerank",
                              f"reranked top-{len(ce_summary)} candidates",
                              ms,
                              {"reranker": getattr(ce, "name", "cross-encoder"),
                               "candidates": ce_summary})

            # ---------------- Evidence + Explanation ----------------
            explainer = get_clinical_explainer(req.explainer)
            diagnoses_out: list[DiagnosisDTO] = []
            evidence_summary: list[dict] = []
            explain_summary: list[dict] = []
            explain_total = 0.0

            t = time.perf_counter()
            cards_per_disease: dict[str, list] = {}
            for cand in candidates:
                cards = cards_for_disease(retr_passages.get(cand.disease, []),
                                              query_symptoms=req.symptoms,
                                              disease_keywords=DISEASE_KEYWORDS.get(cand.disease, []),
                                              max_cards=req.max_evidence_cards)
                if req.passage_type_filter:
                    cards = [c for c in cards if c.passage_type == req.passage_type_filter]
                cards_per_disease[cand.disease] = cards
                evidence_summary.append({
                    "disease": cand.disease,
                    "n_cards": len(cards),
                    "top_card": (
                        {"source": cards[0].source_label,
                         "tier": cards[0].source_tier,
                         "passage_type": cards[0].passage_type,
                         "specificity": round(cards[0].specificity, 3)}
                        if cards else None)})
            ms = (time.perf_counter() - t) * 1000.0
            yield stage("evidence", "Evidence cards",
                          f"{len(candidates)} diagnoses · claim-level extraction · tier sort",
                          ms, {"per_diagnosis": evidence_summary})

            for cand in candidates:
                rules = miner.matching_rules(req.symptoms, cand.disease, top_n=3)
                cards = cards_per_disease[cand.disease]
                t = time.perf_counter()
                try:
                    expl = explainer.explain(disease=cand.disease,
                                                query_symptoms=req.symptoms,
                                                matching_rules=rules,
                                                evidence_cards=cards)
                except Exception as exc:  # noqa: BLE001
                    from src.clinical_explanation import TemplateClinicalExplainer
                    expl = TemplateClinicalExplainer().explain(
                        disease=cand.disease, query_symptoms=req.symptoms,
                        matching_rules=rules, evidence_cards=cards)
                    expl.backend = f"fallback:{type(exc).__name__}"
                explain_total += (time.perf_counter() - t) * 1000.0
                explain_summary.append({
                    "disease": cand.disease, "backend": expl.backend,
                    "preview": expl.symptom_disease_link[:140] + (
                        "…" if len(expl.symptom_disease_link) > 140 else ""),
                    "citation_count": len(expl.citations)})
                diagnoses_out.append(DiagnosisDTO(
                    disease=cand.disease,
                    disease_pretty=cand.disease.replace("_", " ").title(),
                    fused_score=cand.fused_score,
                    mining_score=cand.mining_score,
                    retrieval_score=cand.retrieval_score,
                    matching_rules=[MatchingRuleDTO(**r) for r in rules],
                    evidence_cards=[EvidenceCardDTO(**c.to_dict()) for c in cards],
                    explanation=expl.to_dict()))

            yield stage("explain", "Clinical explanation",
                          f"{explainer.name} · 4-section JSON · {len(candidates)} diagnoses",
                          explain_total,
                          {"backend": explainer.name,
                           "model": getattr(explainer, "model", None),
                           "per_diagnosis": explain_summary})

            timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            related_dto = []
            for p in related[:5]:
                meta = SOURCE_AUTHORITY.get(p.get("source", ""), DEFAULT_SOURCE)
                related_dto.append(RelatedContextDTO(
                    passage_id=str(p["id"]), source=p.get("source", ""),
                    source_label=meta["label"], focus=p.get("focus", ""),
                    question=p.get("question", ""), text=p.get("text", ""),
                    score=float(p.get("score", 0.0))))

            # Push to the live-latency ring buffer (same as /diagnose).
            _LATENCY_HISTORY.append({
                "ts": time.time(),
                "total_ms": timings["total_ms"],
                "encode_ms": timings.get("encode_ms", 0.0),
                "vector_search_ms": timings.get("vector_search_ms", 0.0),
                "mining_ms": timings.get("mining_ms", 0.0),
                "explain_ms": timings.get("explain_ms", 0.0),
                "backend": req.backend, "explainer": req.explainer})
            if len(_LATENCY_HISTORY) > _LATENCY_HISTORY_CAP:
                del _LATENCY_HISTORY[: len(_LATENCY_HISTORY) - _LATENCY_HISTORY_CAP]

            # Final aggregate event with the full DiagnoseResponse shape so
            # the client doesn't need to recompute anything from the stage
            # events. This carries the diagnoses, related_context, and the
            # full pipeline_trace already accumulated in `trace`.
            full = DiagnoseResponse(
                query_symptoms=req.symptoms,
                used_alpha=used_alpha,
                used_backend=("none" if req.mode == "mining-only" else req.backend),
                used_mode=req.mode,
                explainer_backend=explainer.name,
                vector_store=os.environ.get("VECTOR_STORE", "faiss"),
                diagnoses=diagnoses_out,
                related_context=related_dto,
                latency_ms=timings,
                pipeline_trace=[PipelineStageDTO(**s) for s in trace],
            )
            yield emit("complete", full.model_dump())
        except Exception as exc:  # noqa: BLE001
            yield emit("error", {"detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )
