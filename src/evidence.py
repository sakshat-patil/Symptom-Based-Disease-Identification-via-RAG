"""Claim-level evidence extraction from MedQuAD passages.

The proposal asks us to "provide grounded evidence from medical literature
to support those suggestions." Returning a 1000-character paragraph and
calling it evidence isn't grounding -- a clinician needs to know:

    - WHICH sentence in the passage actually supports the diagnosis
    - HOW authoritative the source is (NIH NHLBI > generic Q&A)
    - WHAT TYPE of evidence it is (symptom description vs diagnostic
      criteria vs treatment vs epidemiology)
    - HOW SPECIFIC it is to the patient's symptoms (mentions the actual
      symptoms? just the disease in general?)

This module produces an `EvidenceCard` for every (disease, passage) pair
that the retriever surfaces. The card carries:

    - the highlighted sentence(s) that actually mention the symptoms or
      disease, with character offsets so the UI can highlight them
    - a source authority tier (1=highest, 3=lowest), derived from the
      MedQuAD subdir the passage came from
    - a passage type label (symptoms / diagnosis / treatment / overview)
      inferred from the passage's question text
    - a specificity score: how many of the query symptoms appear in the
      passage text, normalised by query length

The Streamlit/Next.js UI uses these to render real evidence cards instead
of just showing raw text blobs.

Owner: Aishwarya wrote the structure, Vineet contributed the source-tier
mapping and Sakshat reviewed the symptom-mention regex rules.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# MedQuAD ships passages in numbered subdirs by source. We tier them by
# medical authority -- the tier is what the UI shows as a coloured badge.
SOURCE_AUTHORITY: dict[str, dict] = {
    "1_CancerGov_QA":            {"tier": 1, "label": "NCI (Cancer.gov)"},
    "2_GARD_QA":                 {"tier": 1, "label": "NIH GARD"},
    "3_GHR_QA":                  {"tier": 1, "label": "NIH GHR"},
    "4_MPlus_Health_Topics_QA":  {"tier": 1, "label": "NIH MedlinePlus"},
    "5_NIDDK_QA":                {"tier": 1, "label": "NIH NIDDK"},
    "6_NINDS_QA":                {"tier": 1, "label": "NIH NINDS"},
    "7_SeniorHealth_QA":         {"tier": 2, "label": "NIH SeniorHealth"},
    "8_NHLBI_QA_XML":            {"tier": 1, "label": "NIH NHLBI"},
    "9_CDC_QA":                  {"tier": 1, "label": "CDC"},
    "10_MPlus_ADAM_QA":          {"tier": 2, "label": "MedlinePlus A.D.A.M."},
    "11_MPlusDrugs_QA":          {"tier": 2, "label": "MedlinePlus Drugs"},
    "12_MPlusHerbsSupplements_QA":{"tier": 3, "label": "MedlinePlus Herbs"},
}

DEFAULT_SOURCE = {"tier": 3, "label": "MedQuAD"}


# Crude but effective passage-type classifier from the question text.
PASSAGE_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"\bsymptoms?\b|\bsigns?\b|\bpresent(s|ation)?\b", "symptoms"),
    (r"\bdiagn(osis|ose|ostic|osed)\b|\btests?\b", "diagnosis"),
    (r"\btreat(ments|ment|ed)?\b|\btherap(y|ies)\b|\bmanag(ed|ement)\b", "treatment"),
    (r"\bcauses?\b|\brisk\b|\bwhy\b", "causes"),
    (r"\bcomplications?\b|\boutlook\b|\bprognos\w+", "complications"),
    (r"\bprevent(ion|ed)?\b", "prevention"),
    (r"\bgenetic\b|\binherit\w*", "genetics"),
    (r"\bwhat (is|are)\b|\boverview\b", "overview"),
]


@dataclass
class EvidenceClaim:
    """A single highlighted sentence from a passage."""
    sentence: str
    char_start: int
    char_end: int
    matched_terms: list[str]


@dataclass
class EvidenceCard:
    """One passage's evidence for one disease, in clinician-friendly form."""
    passage_id: str
    source_id: str           # raw MedQuAD subdir
    source_label: str        # human label e.g. "NIH NHLBI"
    source_tier: int         # 1 (highest authority) -- 3
    passage_type: str        # symptoms / diagnosis / treatment / ...
    focus: str               # MedQuAD <Focus> = topic of the article
    question: str            # passage question
    full_text: str           # the raw passage text
    claims: list[EvidenceClaim]   # highlighted sentences in the passage
    specificity: float       # fraction of query symptoms mentioned in passage [0..1]
    retrieval_score: float   # cosine sim from the bi-encoder
    ce_score: float | None = None   # cross-encoder score, if available

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# --- helpers ---------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return [(sentence, start, end)] using a conservative regex.

    Not perfect on biomedical abbreviations, but good enough for highlighting.
    """
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in _SENT_SPLIT.split(text):
        if not piece.strip():
            cursor += len(piece) + 1
            continue
        # Locate piece in text starting at cursor.
        idx = text.find(piece, cursor)
        if idx < 0:
            cursor += len(piece) + 1
            continue
        out.append((piece.strip(), idx, idx + len(piece)))
        cursor = idx + len(piece)
    return out


def _classify_passage_type(question: str) -> str:
    q = (question or "").lower()
    for pattern, label in PASSAGE_TYPE_PATTERNS:
        if re.search(pattern, q):
            return label
    return "general"


def _source_meta(source_id: str) -> dict:
    return SOURCE_AUTHORITY.get(source_id, DEFAULT_SOURCE)


def _normalise_for_match(token: str) -> list[str]:
    """Generate candidate strings to look for given a snake_case token.

    `muscle_pain` -> ['muscle pain', 'muscle ache']
    We don't pull in synonyms here; the caller decides whether to expand.
    """
    base = token.replace("_", " ").strip().lower()
    return [base] if base else []


def _find_claims(text: str, query_terms: list[str],
                 disease_keywords: list[str],
                 max_claims: int = 3) -> list[EvidenceClaim]:
    """Pick sentences that mention the query symptoms or disease keywords.

    Ranking: prefer sentences with more matched terms, ties broken by
    earlier position in the passage.
    """
    if not text:
        return []
    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        return []

    # Build a flat list of search terms (already lowercased phrases).
    needles: list[str] = []
    for q in query_terms:
        needles.extend(_normalise_for_match(q))
    for kw in disease_keywords:
        needles.append(kw.lower())
    needles = [n for n in needles if n]
    if not needles:
        return []

    scored: list[tuple[int, int, EvidenceClaim]] = []
    for i, (sent, start, end) in enumerate(sentences):
        low = sent.lower()
        matched = [n for n in needles if n in low]
        if not matched:
            continue
        scored.append((len(matched), -i,
                        EvidenceClaim(sentence=sent, char_start=start,
                                       char_end=end, matched_terms=matched)))
    scored.sort(reverse=True)  # highest match count first, tie -> earliest
    return [c for _, _, c in scored[:max_claims]]


def _specificity(text: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    low = (text or "").lower()
    hits = 0
    for q in query_terms:
        for needle in _normalise_for_match(q):
            if needle and needle in low:
                hits += 1
                break
    return hits / len(query_terms)


# --- public API ------------------------------------------------------------

def build_evidence_card(passage: dict, *, query_symptoms: list[str],
                         disease_keywords: list[str]) -> EvidenceCard:
    """Build one EvidenceCard from a passage hit returned by the retriever."""
    src = passage.get("source", "")
    meta = _source_meta(src)
    full = passage.get("text", "")
    return EvidenceCard(
        passage_id=str(passage.get("id", "")),
        source_id=src,
        source_label=meta["label"],
        source_tier=meta["tier"],
        passage_type=_classify_passage_type(passage.get("question", "")),
        focus=passage.get("focus", ""),
        question=passage.get("question", ""),
        full_text=full,
        claims=_find_claims(full, query_symptoms, disease_keywords),
        specificity=_specificity(full, query_symptoms),
        retrieval_score=float(passage.get("score", 0.0)),
        ce_score=passage.get("ce_score"),
    )


def cards_for_disease(passages: list[dict], *, query_symptoms: list[str],
                       disease_keywords: list[str],
                       max_cards: int = 5) -> list[EvidenceCard]:
    """Build cards for the top-N passages already attributed to one disease.

    Sort key: (source_tier asc, specificity desc, retrieval_score desc)
    so the clinician sees the highest-authority + most-specific passages
    first, regardless of the raw cosine similarity.
    """
    cards = [build_evidence_card(p, query_symptoms=query_symptoms,
                                    disease_keywords=disease_keywords)
              for p in passages]
    cards.sort(key=lambda c: (c.source_tier, -c.specificity, -c.retrieval_score))
    # Drop cards with zero claims and zero specificity -- not useful evidence.
    cards = [c for c in cards if c.claims or c.specificity > 0.0]
    return cards[:max_cards]
