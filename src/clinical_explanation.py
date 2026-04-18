"""Clinical-structured explanation, four parts every clinician expects.

Plain prose summaries don't help a clinician. What they actually want is:

    1. Symptom -> disease link  -- which classic presentation pattern fits
    2. Statistical prior         -- how confident is the data
    3. Evidence quality          -- is the source authoritative; how
                                    specific is the cited passage
    4. What's missing            -- what the system *cannot* tell them
                                    (e.g., "ECG required for definitive Dx")

This module produces a `ClinicalExplanation` with those four fields. Every
field carries a citation list pointing back to evidence cards, so a
clinician can click through. We support two backends:

    - "openai"   : GPT-class model with a strict JSON schema. Best quality,
                    needs OPENAI_API_KEY (and OPENAI_BASE_URL if running
                    against an Azure deployment).
    - "template" : Deterministic rule-based stitcher. Always works.
                    Citation-faithful by construction. Used as the
                    automatic fallback whenever the OpenAI call fails.

The factory function `get_clinical_explainer(kind)` chooses; if `kind`
is "auto", we pick OpenAI if a key is set, otherwise template.

Owner: Aishwarya wrote the structured prompt and the JSON schema, Vineet
plumbed the OpenAI client, Sakshat tuned the template fallback so it
matches the OpenAI output's structure.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from .evidence import EvidenceCard


@dataclass
class ClinicalExplanation:
    """The four-section explanation rendered by the UI."""
    symptom_disease_link: str
    statistical_prior: str
    evidence_quality: str
    whats_missing: str
    citations: list[str] = field(default_factory=list)
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- shared helpers --------------------------------------------------------

def _disease_pretty(name: str) -> str:
    return name.replace("_", " ")


def _short_quote(text: str, max_words: int = 22) -> str:
    """A short verbatim quote for inclusion in the explanation."""
    text = re.sub(r"^Q:.*?A:\s*", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def _citation_str(card: EvidenceCard) -> str:
    return f"[{card.source_label} · {card.passage_id}]"


# --- template explainer ----------------------------------------------------

class TemplateClinicalExplainer:
    """Deterministic citation-faithful stitcher.

    Output mirrors the OpenAI explainer's structure exactly so the UI
    doesn't have to branch on backend.
    """
    name = "template"

    def explain(self, *, disease: str, query_symptoms: list[str],
                 matching_rules: list[dict],
                 evidence_cards: list[EvidenceCard]) -> ClinicalExplanation:
        sym_str = ", ".join(s.replace("_", " ") for s in query_symptoms)
        d_pretty = _disease_pretty(disease)

        # 1. Symptom-disease link
        if matching_rules:
            r = matching_rules[0]
            ante = ", ".join(s.replace("_", " ") for s in r["antecedent"])
            link = (f"The reported symptom cluster ({sym_str}) matches a "
                    f"high-confidence pattern in our training data: "
                    f"{{{ante}}} -> {d_pretty}.")
        else:
            link = (f"The reported symptoms ({sym_str}) are clustered with "
                    f"{d_pretty} via biomedical literature retrieval; no "
                    f"direct association rule fired in this case.")

        # 2. Statistical prior
        if matching_rules:
            r = matching_rules[0]
            prior = (f"Mined association rule confidence is "
                     f"{r['confidence']:.2f} with lift {r['lift']:.1f} on "
                     f"a 4,920-record patient transaction table. Lift > 1 "
                     f"indicates the rule is more predictive than chance.")
        else:
            prior = ("No mined rule fires for this candidate at the chosen "
                     "confidence threshold; this prediction is driven by "
                     "retrieval alone.")

        # 3. Evidence quality
        if evidence_cards:
            tier1 = [c for c in evidence_cards if c.source_tier == 1]
            best = tier1[0] if tier1 else evidence_cards[0]
            ev_summary = (
                f"Top supporting passage from {best.source_label} "
                f"({best.passage_type} content, specificity "
                f"{best.specificity:.2f}). Excerpt: \""
                f"{_short_quote(best.claims[0].sentence) if best.claims else _short_quote(best.full_text)}\""
            )
        else:
            ev_summary = ("No high-specificity evidence passage was found for "
                          "this candidate; the system fell back to "
                          "rule-only support.")

        # 4. What's missing
        missing_bits: list[str] = [
            "This system suggests likely diagnoses; it does not replace "
            "physical examination, lab tests, or imaging.",
        ]
        if not evidence_cards:
            missing_bits.append("Note: no peer-reviewed passage was matched "
                                 "for this candidate; treat with caution.")
        if matching_rules and matching_rules[0]["lift"] < 5:
            missing_bits.append("Lift is modest; co-occurrence may be partly "
                                 "explained by base prevalence.")
        missing = " ".join(missing_bits)

        return ClinicalExplanation(
            symptom_disease_link=link,
            statistical_prior=prior,
            evidence_quality=ev_summary,
            whats_missing=missing,
            citations=[_citation_str(c) for c in evidence_cards[:3]],
            backend=self.name,
        )


# --- OpenAI explainer ------------------------------------------------------

_OPENAI_SYSTEM = """You are a medical decision-support assistant integrated
with a hybrid mining + retrieval pipeline. You produce JSON only -- no
prose outside JSON. Every claim you make must be traceable to either the
provided association rule or to one of the evidence excerpts; do not
invent facts. Tone: concise, clinical, hedged. The reader is a clinician
who already knows medical terminology."""

_OPENAI_USER_TEMPLATE = """Diagnosis under consideration: {disease}
Patient symptoms: {symptoms}

Mined association rule for this diagnosis:
{rule_block}

Top evidence excerpts (with source authority and specificity):
{evidence_block}

Return a JSON object with exactly these string fields, each 1-3 sentences:
- symptom_disease_link: how the reported symptom cluster classically maps
  to this diagnosis. Cite either the rule or an excerpt.
- statistical_prior: what the mining confidence/lift tell us about the
  data-driven prior probability. Use the numbers provided.
- evidence_quality: comment on the source authority and specificity of
  the strongest cited excerpt.
- whats_missing: explicitly state what this system cannot determine
  (labs, imaging, physical exam, etc.) that a clinician would need
  before acting on this suggestion.

Output strictly:
{{
  "symptom_disease_link": "...",
  "statistical_prior": "...",
  "evidence_quality": "...",
  "whats_missing": "..."
}}"""


@dataclass
class OpenAIClinicalExplainer:
    name: str = "openai"
    model: str = "gpt-4o-mini"
    _client: Any = None

    @classmethod
    def load(cls, model: str | None = None) -> "OpenAIClinicalExplainer":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # Prefer explicit `model` arg, then OPENAI_CHAT_MODEL env (set when
        # running against an Azure deployment whose name isn't gpt-4o-mini),
        # then fall back to the public OpenAI default.
        chosen = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        # OpenAI() picks up OPENAI_API_KEY + OPENAI_BASE_URL from env, which
        # is how we route to Azure's v1-compat endpoint.
        return cls(model=chosen, _client=OpenAI())

    def _format_rule(self, matching_rules: list[dict]) -> str:
        if not matching_rules:
            return "(no matching rule)"
        r = matching_rules[0]
        ante = ", ".join(s.replace("_", " ") for s in r["antecedent"])
        return (f"  antecedent: {{{ante}}}\n"
                f"  confidence: {r['confidence']:.3f}\n"
                f"  lift: {r['lift']:.2f}")

    def _format_evidence(self, cards: list[EvidenceCard]) -> str:
        if not cards:
            return "(no high-specificity evidence)"
        out = []
        for c in cards[:3]:
            quote = (_short_quote(c.claims[0].sentence) if c.claims
                       else _short_quote(c.full_text))
            out.append(
                f"- {c.source_label} (tier {c.source_tier}, "
                f"type={c.passage_type}, specificity {c.specificity:.2f}): "
                f"\"{quote}\""
            )
        return "\n".join(out)

    def explain(self, *, disease: str, query_symptoms: list[str],
                 matching_rules: list[dict],
                 evidence_cards: list[EvidenceCard]) -> ClinicalExplanation:
        user = _OPENAI_USER_TEMPLATE.format(
            disease=_disease_pretty(disease),
            symptoms=", ".join(s.replace("_", " ") for s in query_symptoms),
            rule_block=self._format_rule(matching_rules),
            evidence_block=self._format_evidence(evidence_cards),
        )
        # GPT-5-class deployments (e.g. Azure 'gpt-5.3-chat') reject both
        # `max_tokens` (use `max_completion_tokens`) AND any non-default
        # `temperature`. Older models (gpt-4o-mini) still expect both.
        # Try the modern shape, fall back on parameter-shape errors.
        msgs = [
            {"role": "system", "content": _OPENAI_SYSTEM},
            {"role": "user", "content": user},
        ]
        def _attempt(kwargs):
            return self._client.chat.completions.create(
                model=self.model, messages=msgs,
                response_format={"type": "json_object"}, **kwargs,
            )
        # GPT-5 reasoning models burn tokens on internal thought before
        # visible output, and `max_completion_tokens` covers both. 400 was
        # enough for gpt-4o-mini but starves gpt-5; budget generously here.
        kw = {"temperature": 0.2, "max_completion_tokens": 1500}
        try:
            try:
                resp = _attempt(kw)
            except Exception as e1:  # noqa: BLE001
                m1 = str(e1)
                if "temperature" in m1:
                    kw.pop("temperature", None)
                    try:
                        resp = _attempt(kw)
                    except Exception as e2:  # noqa: BLE001
                        m1 = str(e2)
                        if "max_tokens" in m1 or "max_completion_tokens" in m1:
                            resp = _attempt({"max_tokens": 600})
                        else:
                            raise
                elif "max_tokens" in m1 or "max_completion_tokens" in m1:
                    resp = _attempt({"max_tokens": 600, "temperature": 0.2})
                else:
                    raise
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            # Fall back to template if OpenAI fails for any reason at runtime.
            tpl = TemplateClinicalExplainer()
            out = tpl.explain(disease=disease, query_symptoms=query_symptoms,
                                matching_rules=matching_rules,
                                evidence_cards=evidence_cards)
            out.backend = f"openai-failed:{type(exc).__name__}>>template"
            return out

        return ClinicalExplanation(
            symptom_disease_link=data.get("symptom_disease_link", "").strip(),
            statistical_prior=data.get("statistical_prior", "").strip(),
            evidence_quality=data.get("evidence_quality", "").strip(),
            whats_missing=data.get("whats_missing", "").strip(),
            citations=[_citation_str(c) for c in evidence_cards[:3]],
            backend=f"openai:{self.model}",
        )


# --- factory ---------------------------------------------------------------

@lru_cache(maxsize=4)
def get_clinical_explainer(kind: str = "auto"):
    if kind == "auto":
        kind = "openai" if os.environ.get("OPENAI_API_KEY") else "template"
    if kind == "template":
        return TemplateClinicalExplainer()
    if kind == "openai":
        return OpenAIClinicalExplainer.load()
    raise ValueError(f"unknown clinical explainer kind: {kind}")
