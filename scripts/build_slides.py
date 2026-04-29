"""Build the 6-slide CMPE 255 presentation deck (title slide excluded from the
6-slide budget per the assignment).

Slides:
    0. Title
    1. Problem & motivation
    2. System architecture
    3. Data & methodology
    4. Results (ablation + alpha sweep)
    5. UI demo
    6. Conclusion & future work
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT = ROOT.parent / "deliverables" / "slides" / "presentation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(0x0D, 0x3B, 0x66)
ACCENT = RGBColor(0x3A, 0x86, 0xFF)
INK = RGBColor(0x1F, 0x2A, 0x3D)
MUTED = RGBColor(0x6B, 0x72, 0x80)
LIGHT_BG = RGBColor(0xF6, 0xF8, 0xFC)


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK,
              align=None, font="Calibri"):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    if isinstance(text, str):
        runs = [text]
    else:
        runs = text
    for i, t in enumerate(runs):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align is not None:
            para.alignment = align
        run = para.add_run()
        run.text = t
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def add_bullets(slide, x, y, w, h, bullets, size=16, color=INK):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(6)
        run = para.add_run()
        run.text = "•  " + b
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return box


def add_band(slide, color=NAVY, height=0.45):
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                    SLIDE_W, Inches(height))
    band.fill.solid()
    band.fill.fore_color.rgb = color
    band.line.fill.background()


def add_footer(slide, label):
    add_text(slide, Inches(0.4), SLIDE_H - Inches(0.4), Inches(11), Inches(0.3),
              "CMPE 255 Spring 2026  ·  "
              "Sakshat Patil  ·  Vineet Kumar  ·  Aishwarya Madhave  ·  "
              + label,
              size=10, color=MUTED)


def header(slide, title_text, subtitle_text=None):
    add_band(slide)
    add_text(slide, Inches(0.5), Inches(0.55), Inches(12), Inches(0.7),
              title_text, size=26, bold=True, color=NAVY)
    if subtitle_text:
        add_text(slide, Inches(0.5), Inches(1.15), Inches(12), Inches(0.4),
                  subtitle_text, size=14, color=MUTED)


def picture_safe(slide, path: Path, x, y, **kwargs):
    if not path.exists():
        return None
    return slide.shapes.add_picture(str(path), x, y, **kwargs)


def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6),
                                  SLIDE_W, Inches(0.05))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()

    add_text(s, Inches(0.7), Inches(2.9), Inches(12), Inches(1.0),
              "Record-Based Medical Diagnostic Assistant",
              size=42, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_text(s, Inches(0.7), Inches(3.85), Inches(12), Inches(0.6),
              "A Hybrid FP-Growth + Retrieval-Augmented Pipeline",
              size=22, color=RGBColor(0xCB, 0xD5, 0xE1))
    add_text(s, Inches(0.7), Inches(5.2), Inches(12), Inches(0.5),
              "Sakshat Nandkumar Patil (018318287)  ·  "
              "Vineet Kumar (019140433)  ·  "
              "Aishwarya Madhave (019129110)",
              size=16, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_text(s, Inches(0.7), Inches(5.65), Inches(12), Inches(0.4),
              "Department of Computer Engineering  ·  "
              "CMPE 255 Data Mining (Spring 2026)  ·  San José State University",
              size=13, color=RGBColor(0xCB, 0xD5, 0xE1))


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "1.  Problem & Motivation",
           "Why diagnostic decision support needs both rules and evidence")
    add_bullets(s, Inches(0.6), Inches(1.7), Inches(7.5), Inches(5.0), [
        "Clinicians face cognitive overload synthesizing symptoms, history, and the medical literature.",
        "Existing tools either rely on hand-built ontologies (expensive) or are black-box models (no explanation).",
        "Goal: surface ranked diagnoses from a symptom set AND ground each one in citable evidence.",
        "Our angle: combine interpretable association rules with a biomedical retrieval index.",
        "Every prediction in our system is auditable through both a mined rule and the retrieved passage.",
    ], size=18)

    # Side panel: key numbers
    panel = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(8.4), Inches(1.7), Inches(4.4),
                                  Inches(5.0))
    panel.fill.solid()
    panel.fill.fore_color.rgb = LIGHT_BG
    panel.line.color.rgb = ACCENT
    panel.line.width = Pt(1.5)

    add_text(s, Inches(8.6), Inches(1.85), Inches(4.0), Inches(0.45),
              "By the numbers", size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(8.6), Inches(2.4), Inches(4.0), Inches(4.5), [
        "4,920 patient records",
        "41 diseases, 131 symptom tokens",
        "23,839 association rules mined",
        "24,063 MedQuAD passages indexed",
        "200 synthetic test cases",
        "82.5% Recall@1, 0.857 MRR",
    ], size=15)
    add_footer(s, "Slide 1 / 6")


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "2.  System Architecture",
           "All six proposal steps wired together")

    def block(x_in, y_in, w_in, h_in, color, label_top, body):
        x, y = Inches(x_in), Inches(y_in)
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y,
                                Inches(w_in), Inches(h_in))
        b.fill.solid()
        b.fill.fore_color.rgb = LIGHT_BG
        b.line.color.rgb = color
        b.line.width = Pt(2)
        add_text(s, x, y + Inches(0.10), Inches(w_in), Inches(0.45),
                  label_top, size=14, bold=True, color=color, align=1)
        add_bullets(s, x + Inches(0.20), y + Inches(0.6),
                     Inches(w_in - 0.4), Inches(h_in - 0.7),
                     body, size=11)

    # Top row: data + mining
    block(0.3, 1.85, 4.1, 2.3, RGBColor(0x6B, 0x72, 0x80),
           "Step 1 · Data ETL",
           ["Kaggle CSV → 4,920 transactions",
            "Synthea FHIR ETL (parallel path)",
            "131 unique symptom tokens",
            "Sakshat — owner"])
    block(4.6, 1.85, 4.1, 2.3, ACCENT,
           "Step 2 · FP-Growth Mining",
           ["mlxtend, min_support=0.005",
            "23,839 rules / 41 diseases",
            "MiningConf(Q,d) at query time",
            "Sakshat — owner"])
    block(8.9, 1.85, 4.1, 2.3, RGBColor(0x10, 0xB9, 0x81),
           "Step 3 · Knowledge Base",
           ["24,063 MedQuAD passages",
            "MiniLM 384d / PubMedBERT 768d / Azure 3072d",
            "FAISS local · Pinecone serverless",
            "Vineet — owner"])

    # Bottom row: RAG + fusion + eval
    block(0.3, 4.3, 4.1, 2.3, RGBColor(0xEF, 0x47, 0x6F),
           "Step 4 · RAG Pipeline",
           ["Dense retrieval + synonym expansion",
            "Pinecone reranker: Cohere 3.5 / BGE-v2-m3",
            "Template + OpenAI structured explainers",
            "Vineet + Aishwarya"])
    block(4.6, 4.3, 4.1, 2.3, RGBColor(0xF6, 0x9B, 0x2C),
           "Step 5 · Hybrid Fusion",
           ["α·RetrievalSim + (1−α)·MiningConf",
            "α = 0.3 (sweep-validated)",
            "Pooled candidate set",
            "Aishwarya — owner"])
    block(8.9, 4.3, 4.1, 2.3, RGBColor(0x7C, 0x3A, 0xED),
           "Step 6 · Evaluation",
           ["Recall@K, MRR (200 cases)",
            "α sweep + per-stage latency",
            "155 pytest tests (147 unit · 8 live)",
            "Aishwarya — owner"])

    add_text(s, Inches(0.6), Inches(6.75), Inches(12.0), Inches(0.4),
              "Three-tier deployment:  Next.js 14 web  →  FastAPI AI service "
              "(MPS)  →  Data tier (FAISS local | Pinecone managed)",
              size=12, bold=True, color=NAVY, align=1)
    add_text(s, Inches(0.6), Inches(7.10), Inches(12.0), Inches(0.4),
              "Production demo: Azure OpenAI 3072d  →  Pinecone "
              "(255-data-mining, 24,063 vectors)  →  fused mining + retrieval",
              size=11, color=MUTED, align=1)
    add_footer(s, "Slide 2 / 6")


def slide_data(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "3.  Data & Methodology",
           "Two complementary data sources, one fusion rule")

    add_text(s, Inches(0.5), Inches(1.7), Inches(6), Inches(0.4),
              "Disease-Symptom transactions", size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(2.1), Inches(6), Inches(2.5), [
        "4,920 rows × 41 diseases (Kaggle-style schema)",
        "Up to 17 symptoms per row, 131 unique tokens",
        "ETL: lowercase + snake_case normalisation",
        "FP-Growth basket = symptoms + DX:disease item",
    ], size=14)

    add_text(s, Inches(0.5), Inches(4.5), Inches(6), Inches(0.4),
              "MedQuAD biomedical Q&A", size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(4.9), Inches(6), Inches(2.0), [
        "24,063 passages from 11k NIH XML files",
        "1000-char chunks with 200-char overlap",
        "Embedded with MiniLM and PubMedBERT (MPS)",
        "Indexed with FAISS IndexFlatIP",
    ], size=14)

    # Right: equation + synonym example
    add_text(s, Inches(7.0), Inches(1.7), Inches(6), Inches(0.4),
              "Mining-time scoring", size=18, bold=True, color=NAVY)
    add_text(s, Inches(7.0), Inches(2.1), Inches(6), Inches(0.5),
              "MiningConf(Q,d) = max  c(A→d) · |A∩Q|/|A|",
              size=16, color=INK, font="Cambria Math")
    add_text(s, Inches(7.0), Inches(2.55), Inches(6), Inches(0.4),
              "        over rules A→d with A ⊆ Q", size=13, color=MUTED)

    add_text(s, Inches(7.0), Inches(3.4), Inches(6), Inches(0.4),
              "Synonym bridge", size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(7.0), Inches(3.8), Inches(6), Inches(2.5), [
        "muscle_pain  →  myalgia",
        "high_fever  →  pyrexia, hyperthermia",
        "breathlessness  →  dyspnea",
        "yellowish_skin  →  jaundice, icterus",
        "fast_heart_rate  →  tachycardia",
    ], size=14)
    add_text(s, Inches(7.0), Inches(6.0), Inches(6), Inches(0.4),
              "130 entries; covers all 41 disease classes",
              size=12, color=MUTED)
    add_footer(s, "Slide 3 / 6")


def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "4.  Results: 200 synthetic test cases",
           "Fusion beats both signals; α=0.3 is the sweet spot")

    rows = [
        ["Variant", "Mode", "R@1", "R@10", "MRR"],
        ["mining-only", "α=0.0", "0.790", "0.870", "0.829"],
        ["MiniLM", "retrieval", "0.130", "0.180", "0.151"],
        ["MiniLM+syn", "fused 0.3", "0.820", "0.895", "0.857"],
        ["PubMedBERT+syn", "retrieval", "0.130", "0.265", "0.194"],
        ["MiniLM   (best)", "fused 0.3", "0.825", "0.890", "0.857"],
    ]
    table_shape = s.shapes.add_table(len(rows), len(rows[0]),
                                       Inches(0.5), Inches(1.7),
                                       Inches(6.5), Inches(2.8))
    tbl = table_shape.table
    for c in range(len(rows[0])):
        tbl.columns[c].width = Inches([2.2, 1.4, 0.9, 0.9, 0.9][c])
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = v
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(13)
                    run.font.bold = (i == 0)
                    run.font.color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if i == 0
                                          else INK)
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY

    add_text(s, Inches(0.5), Inches(4.7), Inches(6.5), Inches(0.4),
              "Best fused: MiniLM + synonym expansion, α=0.3",
              size=14, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(5.05), Inches(6.5), Inches(1.6), [
        "+3.5 R@1 over mining alone",
        "Retrieval-only collapses (vocab gap); +syn helps but ceiling stays low",
        "Signals are complementary, not redundant",
    ], size=13)
    # Latency callout
    panel = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(0.5), Inches(6.4),
                                  Inches(6.5), Inches(0.7))
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(0xEC, 0xFE, 0xFF)
    panel.line.color.rgb = RGBColor(0x0E, 0x74, 0x90)
    panel.line.width = Pt(1.2)
    add_text(s, Inches(0.7), Inches(6.5), Inches(6.2), Inches(0.5),
              "⏱ Latency: 12 ms mean / 20 ms p95 (default config, "
              "100 queries on M3 Pro)",
              size=12, bold=True, color=RGBColor(0x0E, 0x74, 0x90))

    picture_safe(s, RESULTS / "alpha_sweep.png", Inches(7.4), Inches(1.7),
                  width=Inches(5.6))
    add_text(s, Inches(7.4), Inches(6.4), Inches(5.6), Inches(0.4),
              "α-sweep peaks on [0.1, 0.4] — Recall@1 collapses for α≥0.7.",
              size=11, color=MUTED, align=1)
    add_footer(s, "Slide 4 / 6")


def slide_demo(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "5.  Live Demo: Next.js Clinical UI",
           "Grounded evidence the proposal asked for")

    add_bullets(s, Inches(0.5), Inches(1.7), Inches(6.5), Inches(5.2), [
        "Next.js 14 web app talks to FastAPI inference service over REST.",
        "Symptom picker with autocomplete + 6 clinical presets.",
        "Per-diagnosis card has 4 sections: symptom→disease link, statistical prior, evidence quality, what's missing.",
        "Evidence cards show the highlighted sentence (yellow) that actually mentions the symptom or disease.",
        "Source authority badges (NIH NHLBI = tier 1, etc.) and passage-type chips (symptoms / diagnosis / treatment).",
        "Live α slider, mode/backend toggles, source + passage-type filters.",
        "OpenAI gpt-4o-mini explainer toggle (optional, with template fallback).",
        "Latency strip after every query: total + per-stage breakdown.",
    ], size=13)

    panel = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(7.5), Inches(1.7),
                                  Inches(5.4), Inches(5.0))
    panel.fill.solid()
    panel.fill.fore_color.rgb = NAVY
    panel.line.fill.background()
    add_text(s, Inches(7.7), Inches(1.85), Inches(5.0), Inches(0.4),
              "Demo flow", size=16, bold=True,
              color=RGBColor(0xFF, 0xFF, 0xFF))
    add_bullets(s, Inches(7.7), Inches(2.3), Inches(5.0), Inches(4.2), [
        "1. Pick 'Cardiac event' preset.",
        "2. heart_attack ranks #1 (mining 1.00, fused 0.70).",
        "3. Read clinical card: link / prior / evidence / what's missing.",
        "4. Inspect highlighted MedQuAD sentence with NIH NHLBI badge.",
        "5. Filter to source = MedlinePlus → see only those passages.",
        "6. Toggle to OpenAI explainer → richer prose, same citations.",
        "7. Drag α slider: rankings re-fuse live.",
    ], size=13, color=RGBColor(0xE2, 0xE8, 0xF0))
    add_footer(s, "Slide 5 / 6")


def slide_conclusion(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "6.  Conclusion & Future Work",
           "What we shipped, what's next")

    add_text(s, Inches(0.5), Inches(1.7), Inches(6), Inches(0.4),
              "Shipped", size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(2.2), Inches(6), Inches(4.5), [
        "Reproducible FP-Growth pipeline (23,839 rules, 100% disease coverage)",
        "Three encoders: MiniLM 384d, PubMedBERT 768d, Azure OpenAI 3072d",
        "Pinecone Serverless production index (255-data-mining, 24,063 vectors)",
        "Claim-level evidence with NIH source-tier badges + specificity",
        "Pinecone Cohere/BGE reranker with auto-fallback (free tier safe)",
        "Clinical-structured explainer: deterministic template + OpenAI",
        "Next.js 14 UI + FastAPI microservice + Python data tier",
        "82.5% R@1, 0.857 MRR · Heart Attack fused 0.831 on Azure path",
        "155 pytest tests passing (147 unit, 8 live integration)",
    ], size=13)

    add_text(s, Inches(7.0), Inches(1.7), Inches(6), Inches(0.4),
              "Future Work", size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(7.0), Inches(2.2), Inches(6), Inches(4.5), [
        "UMLS-backed automatic synonym mining (replace curated dict)",
        "Cross-encoder reranker over <disease, passage> pairs",
        "Evaluate on Synthea FHIR pipeline (held-out vocabulary)",
        "Open-domain disease universe via UMLS canonicalisation",
        "Learned fusion (XGBoost) using rule lift, passage source, overlap",
    ], size=14)

    add_text(s, Inches(0.5), Inches(6.45), Inches(12), Inches(0.4),
              "Sakshat Patil  ·  Vineet Kumar  ·  Aishwarya Madhave",
              size=13, bold=True, color=NAVY, align=1)
    add_text(s, Inches(0.5), Inches(6.85), Inches(12), Inches(0.4),
              "Code: github.com/sakshat-patil/Symptom-Based-Disease-Identification-via-RAG",
              size=11, color=MUTED, align=1)
    add_footer(s, "Slide 6 / 6")


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs)
    slide_data(prs)
    slide_results(prs)
    slide_demo(prs)
    slide_conclusion(prs)

    prs.save(str(OUT))
    print(f"[slides] wrote {OUT}")


if __name__ == "__main__":
    build()
