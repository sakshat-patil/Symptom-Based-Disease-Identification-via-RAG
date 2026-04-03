"""
MedQuAD preprocessor — converts MedQuAD XML files into a JSONL passage corpus
for the dense retrieval pipeline.

MedQuAD structure:
    data/raw/MedQuAD/
        1_CancerGov_QA/  *.xml
        2_GARD_QA/       *.xml
        ...              (12 source folders, 11,274 XML files total)

Each XML file has:
    <Document source="..." url="...">
      <Focus>Disease Name</Focus>
      <QAPairs>
        <QAPair pid="1">
          <Question qtype="symptoms">...</Question>
          <Answer>...</Answer>
        </QAPair>
        ...
      </QAPairs>
    </Document>

Output: data/raw/passages.jsonl
    Each line: {"text": "...", "disease": "...", "source": "...", "qtype": "..."}

Usage:
    python src/medquad_preprocessor.py
    python src/medquad_preprocessor.py --medquad_dir data/raw/MedQuAD \
        --out data/raw/passages.jsonl --max_chars 1000 --qtypes symptoms information treatment
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# QA types to include as passages (all by default)
# ---------------------------------------------------------------------------
ALL_QTYPES = {
    "symptoms", "information", "treatment", "causes", "outlook",
    "susceptibility", "prevention", "research", "exams and tests",
    "stages", "inheritance",
}

# QA types most relevant to symptom-based diagnosis — used as default filter
DIAGNOSTIC_QTYPES = {"symptoms", "information", "causes", "susceptibility"}


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_medquad_xml(filepath: Path) -> list[dict]:
    """Parse one MedQuAD XML file and return a list of passage dicts.

    Each passage corresponds to one QAPair and contains:
        text     — concatenation of question + answer (chunked to max_chars later)
        disease  — content of <Focus> element
        source   — document source attribute (e.g. "CancerGov")
        qtype    — question type attribute (e.g. "symptoms")
        url      — source URL

    Returns empty list on parse errors.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return []

    source = root.attrib.get("source", "")
    url    = root.attrib.get("url", "")

    focus_el = root.find("Focus")
    disease  = _clean_text(focus_el.text) if focus_el is not None else ""

    passages = []
    for qa_pair in root.findall(".//QAPair"):
        q_el = qa_pair.find("Question")
        a_el = qa_pair.find("Answer")

        question = _clean_text(q_el.text)  if q_el is not None else ""
        answer   = _clean_text(a_el.text)  if a_el is not None else ""
        qtype    = (q_el.attrib.get("qtype", "") if q_el is not None else "").lower()

        if not answer:
            continue

        # Build passage text: "Q: <question> A: <answer>"
        text = f"{question} {answer}".strip() if question else answer

        passages.append({
            "text":    text,
            "disease": disease,
            "source":  source,
            "qtype":   qtype,
            "url":     url,
        })

    return passages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_passage(passage: dict, max_chars: int = 1000, overlap_chars: int = 100) -> list[dict]:
    """Split a long passage into overlapping chunks of at most max_chars characters.

    Short passages (≤ max_chars) are returned as-is.

    Parameters
    ----------
    passage:
        Dict with at least a ``"text"`` key.
    max_chars:
        Maximum character length per chunk.
    overlap_chars:
        Number of characters of overlap between consecutive chunks.

    Returns
    -------
    list[dict]  — one or more passage dicts with updated ``"text"``.
    """
    text = passage["text"]
    if len(text) <= max_chars:
        return [passage]

    chunks = []
    start  = 0
    step   = max_chars - overlap_chars

    while start < len(text):
        end   = min(start + max_chars, len(text))
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > max_chars // 2:
                chunk = chunk[:last_period + 1]
                end   = start + last_period + 1

        new_passage = dict(passage)
        new_passage["text"] = chunk.strip()
        if new_passage["text"]:
            chunks.append(new_passage)

        start += step
        if start >= end:
            break

    return chunks or [passage]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_medquad(
    medquad_dir: str | Path = "data/raw/MedQuAD",
    out_path:    str | Path = "data/raw/passages.jsonl",
    qtypes:      set[str] | None = None,
    max_chars:   int = 1000,
    min_chars:   int = 50,
) -> int:
    """Parse all MedQuAD XML files and write a JSONL passage corpus.

    Parameters
    ----------
    medquad_dir:
        Root directory of the cloned MedQuAD repo.
    out_path:
        Output JSONL file path.
    qtypes:
        Set of question types to keep.  ``None`` → keep all types.
    max_chars:
        Maximum passage length in characters (longer passages are chunked).
    min_chars:
        Minimum passage length — shorter passages are dropped.

    Returns
    -------
    int  — total number of passages written.
    """
    medquad_dir = Path(medquad_dir)
    out_path    = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not medquad_dir.exists():
        raise FileNotFoundError(
            f"MedQuAD directory not found at '{medquad_dir}'.\n"
            "Clone it with: git clone https://github.com/abachaa/MedQuAD data/raw/MedQuAD"
        )

    xml_files = sorted(medquad_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under '{medquad_dir}'.")

    print(f"[medquad] Found {len(xml_files)} XML files in '{medquad_dir}'.")
    if qtypes:
        print(f"[medquad] Filtering to qtypes: {sorted(qtypes)}")

    n_written  = 0
    n_skipped  = 0
    seen_texts = set()   # deduplicate identical passages

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, xml_path in enumerate(xml_files):
            passages = parse_medquad_xml(xml_path)

            for p in passages:
                # Filter by qtype
                if qtypes and p["qtype"] not in qtypes:
                    continue

                # Chunk long passages
                chunks = chunk_passage(p, max_chars=max_chars)

                for chunk in chunks:
                    text = chunk["text"]

                    # Drop too-short passages
                    if len(text) < min_chars:
                        n_skipped += 1
                        continue

                    # Deduplicate
                    key = text[:200]
                    if key in seen_texts:
                        n_skipped += 1
                        continue
                    seen_texts.add(key)

                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_written += 1

            if (i + 1) % 1000 == 0:
                print(f"[medquad]   … {i + 1}/{len(xml_files)} files processed "
                      f"({n_written} passages so far)")

    print(f"[medquad] Done. {n_written} passages written to '{out_path}' "
          f"({n_skipped} skipped).")
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MedQuAD XML files into a JSONL passage corpus."
    )
    parser.add_argument(
        "--medquad_dir", default="data/raw/MedQuAD",
        help="Path to the cloned MedQuAD repository.",
    )
    parser.add_argument(
        "--out", default="data/raw/passages.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--qtypes", nargs="*", default=None,
        help="Question types to keep (e.g. symptoms information treatment). "
             "Default: all types.",
    )
    parser.add_argument(
        "--max_chars", type=int, default=1000,
        help="Maximum passage length in characters (longer passages are chunked).",
    )
    parser.add_argument(
        "--min_chars", type=int, default=50,
        help="Minimum passage length — shorter passages are dropped.",
    )
    parser.add_argument(
        "--diagnostic_only", action="store_true",
        help="Shortcut: keep only symptoms/information/causes/susceptibility qtypes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    args = _parse_args()

    print("=" * 60)
    print("MedQuAD Preprocessor")
    print("=" * 60)

    qtypes = None
    if args.diagnostic_only:
        qtypes = DIAGNOSTIC_QTYPES
        print(f"[medquad] --diagnostic_only: keeping {sorted(qtypes)}")
    elif args.qtypes:
        qtypes = set(args.qtypes)

    n = process_medquad(
        medquad_dir=args.medquad_dir,
        out_path=args.out,
        qtypes=qtypes,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )

    # Show a quick sample
    out_path = Path(args.out)
    if out_path.exists() and n > 0:
        print("\nSample passages:")
        with open(out_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                p = json.loads(line)
                print(f"\n  [{i+1}] disease={p['disease']} | qtype={p['qtype']}")
                print(f"       {p['text'][:200]} …")
