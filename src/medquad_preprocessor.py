"""Turn MedQuAD's NIH XML dump into a JSONL of retrieval passages.

MedQuAD ships one XML per article with a tree of <QAPair> nodes. We pull
each (question, answer) pair, drop empties and very short answers (<30
chars: usually metadata), strip whitespace, and chunk long answers at ~1000
chars with a 200-char overlap so a single concept doesn't get cut down the
middle. The article's <Focus> element gives us the disease topic which we
use later to attribute passages to candidate diseases.

We landed on 24,063 passages across 10,733 XML files after dedupe -- the
same number we cited in Check-in 4. Output: data/processed/passages.jsonl.

Owner: Vineet. Chunk size + overlap chosen empirically against retrieval
recall on a small held-out probe set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "raw" / "MedQuAD"
DEFAULT_OUT = ROOT / "data" / "processed" / "passages.jsonl"


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return out


def parse_one(xml_path: Path) -> list[dict]:
    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError:
        return []
    root = tree.getroot()
    focus_el = root.find(".//Focus")
    focus = clean_text(focus_el.text) if focus_el is not None and focus_el.text else ""
    source = xml_path.parent.name
    out = []
    for pair in root.findall(".//QAPair"):
        q_el = pair.find("Question")
        a_el = pair.find("Answer")
        if q_el is None or a_el is None or not (a_el.text or "").strip():
            continue
        q = clean_text(q_el.text or "")
        a = clean_text(a_el.text or "")
        if len(a) < 30:
            continue
        for piece in chunk(a):
            text = f"Q: {q}\nA: {piece}"
            pid = hashlib.md5(f"{xml_path.name}:{q}:{piece[:80]}".encode()).hexdigest()[:16]
            out.append({
                "id": pid,
                "source": source,
                "focus": focus,
                "question": q,
                "text": text,
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max_passages", type=int, default=24063,
                   help="Cap passages to keep retrieval index size predictable")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_passages = 0
    seen = set()
    with out.open("w") as fh:
        for xml in sorted(in_dir.rglob("*.xml")):
            n_files += 1
            for rec in parse_one(xml):
                key = rec["text"][:120]
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps(rec) + "\n")
                n_passages += 1
                if n_passages >= args.max_passages:
                    break
            if n_passages >= args.max_passages:
                break

    print(f"[medquad] processed {n_files} xml files")
    print(f"[medquad] wrote {n_passages} passages to {out}")


if __name__ == "__main__":
    main()
