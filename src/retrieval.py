"""
Dense retrieval pipeline using FAISS and sentence-transformers.

Encodes a corpus of medical passages into a FAISS flat-IP index (cosine
similarity after L2-normalisation) and exposes a retrieve() function that
returns the top-K passages for a free-text symptom query.

Usage (standalone demo):
    python src/retrieval.py
    python src/retrieval.py --corpus data/raw/passages.jsonl --query "fever headache" --top_k 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Hard-coded demo corpus – used when no JSONL file is provided or found.
# Each entry maps to one "passage" that the index will store.
# ---------------------------------------------------------------------------
DEMO_CORPUS = [
    {"text": "Influenza (flu) causes fever, chills, muscle aches, cough, headache, and fatigue. Most patients recover within one to two weeks without requiring medical treatment.", "disease": "Influenza"},
    {"text": "Type 2 diabetes is characterised by high blood sugar, increased thirst, frequent urination, fatigue, and blurred vision. It is managed with lifestyle changes and medication.", "disease": "Type 2 Diabetes"},
    {"text": "Hypertension often presents with no symptoms, but severe cases may cause headache, shortness of breath, and nosebleeds. Long-term management involves antihypertensives.", "disease": "Hypertension"},
    {"text": "Pneumonia symptoms include cough with phlegm, fever, chills, and difficulty breathing. Bacterial pneumonia is treated with antibiotics.", "disease": "Pneumonia"},
    {"text": "Asthma is a chronic respiratory condition marked by wheezing, shortness of breath, chest tightness, and coughing. Inhaled bronchodilators provide rapid relief.", "disease": "Asthma"},
    {"text": "Myocardial infarction (heart attack) presents with chest pain, shortness of breath, nausea, lightheadedness, and cold sweats. It is a medical emergency.", "disease": "Myocardial Infarction"},
    {"text": "Urinary tract infections cause burning urination, frequent urge to urinate, cloudy urine, and pelvic pain. They are commonly treated with antibiotics.", "disease": "Urinary Tract Infection"},
    {"text": "Migraine headaches are severe, pulsating, often unilateral, and may be accompanied by nausea, vomiting, and sensitivity to light and sound.", "disease": "Migraine"},
    {"text": "Chronic obstructive pulmonary disease (COPD) causes persistent cough, mucus production, wheezing, and shortness of breath, especially with exertion.", "disease": "COPD"},
    {"text": "Rheumatoid arthritis is an autoimmune disease causing joint pain, swelling, stiffness, and fatigue, particularly in the morning.", "disease": "Rheumatoid Arthritis"},
    {"text": "Appendicitis presents with pain that starts around the navel and moves to the lower right abdomen, accompanied by nausea, vomiting, and fever.", "disease": "Appendicitis"},
    {"text": "Hypothyroidism symptoms include fatigue, weight gain, cold intolerance, constipation, dry skin, and depression. TSH levels confirm the diagnosis.", "disease": "Hypothyroidism"},
    {"text": "Gastroesophageal reflux disease (GERD) causes heartburn, acid regurgitation, chest pain, and difficulty swallowing. Proton pump inhibitors reduce acid.", "disease": "GERD"},
    {"text": "Depression is characterised by persistent sadness, loss of interest, fatigue, changes in appetite and sleep, and difficulty concentrating.", "disease": "Depression"},
    {"text": "Anxiety disorder presents with excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbance.", "disease": "Anxiety Disorder"},
    {"text": "Celiac disease triggers an immune response to gluten, causing diarrhoea, bloating, abdominal pain, fatigue, and malabsorption of nutrients.", "disease": "Celiac Disease"},
    {"text": "Chronic kidney disease leads to fatigue, decreased urine output, swelling in ankles, shortness of breath, nausea, and confusion in advanced stages.", "disease": "Chronic Kidney Disease"},
    {"text": "Liver cirrhosis symptoms include jaundice, fatigue, abdominal swelling, easy bruising, spider angiomata, and confusion due to hepatic encephalopathy.", "disease": "Liver Cirrhosis"},
    {"text": "Sepsis is a life-threatening response to infection characterised by high fever, rapid heart rate, rapid breathing, and confusion.", "disease": "Sepsis"},
    {"text": "Stroke presents with sudden numbness or weakness of the face, arm, or leg, confusion, trouble speaking, vision problems, and severe headache.", "disease": "Stroke"},
    {"text": "Anemia causes fatigue, pallor, shortness of breath, dizziness, cold hands and feet, and irregular heartbeat due to low red blood cell count.", "disease": "Anemia"},
    {"text": "Osteoporosis weakens bones, increasing fracture risk. It is largely asymptomatic until a fracture occurs, often in the hip, spine, or wrist.", "disease": "Osteoporosis"},
    {"text": "Parkinson's disease is a neurodegenerative disorder causing tremors, bradykinesia, rigidity, and postural instability.", "disease": "Parkinson's Disease"},
    {"text": "Multiple sclerosis (MS) causes fatigue, difficulty walking, numbness, muscle weakness, blurred vision, and problems with coordination.", "disease": "Multiple Sclerosis"},
    {"text": "Lupus is an autoimmune disease presenting with a butterfly rash, joint pain, fatigue, fever, hair loss, and kidney involvement.", "disease": "Lupus"},
    {"text": "Tuberculosis causes persistent cough (sometimes with blood), night sweats, weight loss, fever, and fatigue.", "disease": "Tuberculosis"},
    {"text": "HIV/AIDS impairs the immune system, leading to opportunistic infections, weight loss, fever, night sweats, and fatigue.", "disease": "HIV/AIDS"},
    {"text": "Psoriasis causes red, itchy, scaly plaques on the skin. It is a chronic autoimmune condition with periods of flares and remission.", "disease": "Psoriasis"},
    {"text": "Eczema (atopic dermatitis) causes dry, itchy, inflamed skin. It often begins in childhood and is associated with asthma and hay fever.", "disease": "Eczema"},
    {"text": "Gout is caused by uric acid crystal deposition in joints, causing sudden, severe joint pain, redness, and swelling, most commonly in the big toe.", "disease": "Gout"},
    {"text": "Pancreatitis causes severe upper abdominal pain, nausea, vomiting, and fever. Chronic pancreatitis may lead to malabsorption and diabetes.", "disease": "Pancreatitis"},
    {"text": "Gallstones can cause biliary colic — intermittent upper right abdominal pain, nausea, and vomiting, especially after fatty meals.", "disease": "Gallstones"},
    {"text": "Endometriosis causes pelvic pain, dysmenorrhoea, dyspareunia, and infertility. Diagnosis is confirmed by laparoscopy.", "disease": "Endometriosis"},
    {"text": "Polycystic ovary syndrome (PCOS) causes irregular periods, excess androgen, polycystic ovaries, weight gain, and insulin resistance.", "disease": "PCOS"},
    {"text": "Irritable bowel syndrome (IBS) presents with recurrent abdominal pain, bloating, diarrhoea, and constipation without structural cause.", "disease": "IBS"},
    {"text": "Crohn's disease causes abdominal pain, severe diarrhoea, fatigue, weight loss, and malnutrition, affecting any part of the digestive tract.", "disease": "Crohn's Disease"},
    {"text": "Ulcerative colitis is an inflammatory bowel disease causing bloody diarrhoea, abdominal cramping, and urgency, limited to the colon.", "disease": "Ulcerative Colitis"},
    {"text": "Dengue fever presents with sudden high fever, severe headache, pain behind the eyes, muscle and joint pain, rash, and mild bleeding.", "disease": "Dengue Fever"},
    {"text": "Malaria causes cyclical fever, chills, sweating, headache, nausea, and vomiting. Plasmodium falciparum can cause severe, life-threatening disease.", "disease": "Malaria"},
    {"text": "Chickenpox (varicella) causes an itchy blister rash, fever, tiredness, and loss of appetite. The varicella-zoster virus is highly contagious.", "disease": "Chickenpox"},
    {"text": "Measles causes fever, cough, runny nose, red eyes, and a characteristic maculopapular rash that spreads from the head downwards.", "disease": "Measles"},
    {"text": "COVID-19 symptoms range from fever, cough, and fatigue to severe respiratory distress. Loss of taste and smell is a distinctive symptom.", "disease": "COVID-19"},
    {"text": "Fibromyalgia causes widespread musculoskeletal pain, fatigue, sleep problems, and cognitive difficulties ('fibro fog').", "disease": "Fibromyalgia"},
    {"text": "Peripheral neuropathy causes weakness, numbness, and pain, usually in the hands and feet, due to peripheral nerve damage.", "disease": "Peripheral Neuropathy"},
    {"text": "Glaucoma is characterised by progressive optic nerve damage, often associated with elevated intraocular pressure, leading to vision loss.", "disease": "Glaucoma"},
    {"text": "Cataracts cause cloudy or blurred vision, faded colours, poor night vision, and glare sensitivity due to lens opacification.", "disease": "Cataracts"},
    {"text": "Prostate cancer may cause urinary symptoms such as weak flow, frequency, urgency, and nocturia, or may be asymptomatic in early stages.", "disease": "Prostate Cancer"},
    {"text": "Breast cancer may present as a breast lump, nipple discharge, skin dimpling, or axillary lymph node enlargement.", "disease": "Breast Cancer"},
    {"text": "Lung cancer causes persistent cough, haemoptysis, chest pain, hoarseness, weight loss, and recurrent respiratory infections.", "disease": "Lung Cancer"},
    {"text": "Colorectal cancer symptoms include changes in bowel habits, rectal bleeding, abdominal pain, weight loss, and fatigue.", "disease": "Colorectal Cancer"},
]


# ---------------------------------------------------------------------------
# Lazy model loader — avoids importing sentence_transformers at module level
# so the file is importable even without the dependency installed.
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"[retrieval] Loading encoder: {_MODEL_NAME}")
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(jsonl_path: str | Path | None) -> list[dict]:
    """Load passages from a JSONL file.  Falls back to DEMO_CORPUS if missing.

    Each line must contain at least a ``"text"`` key.  An optional
    ``"disease"`` key is preserved for display / evaluation.

    Parameters
    ----------
    jsonl_path:
        Path to the ``.jsonl`` file, or ``None`` to always use the demo corpus.

    Returns
    -------
    list[dict]
        List of passage dicts, each with at least ``{"text": str}``.
    """
    if jsonl_path is None:
        print("[retrieval] No corpus path given — using built-in demo corpus.")
        return DEMO_CORPUS

    path = Path(jsonl_path)
    if not path.exists():
        print(f"[retrieval] '{path}' not found — using built-in demo corpus.")
        return DEMO_CORPUS

    passages = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    print(f"[retrieval] Loaded {len(passages)} passages from '{path}'.")
    return passages


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def encode_passages(passages: list[dict], batch_size: int = 64) -> np.ndarray:
    """Encode a list of passage dicts into L2-normalised float32 embeddings.

    Parameters
    ----------
    passages:
        Each dict must have a ``"text"`` key.
    batch_size:
        Sentence-transformer encoding batch size.

    Returns
    -------
    np.ndarray, shape (n_passages, dim)
    """
    model = _get_model()
    texts = [p["text"] for p in passages]
    print(f"[retrieval] Encoding {len(texts)} passages …")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string into a L2-normalised float32 vector.

    Returns
    -------
    np.ndarray, shape (1, dim)
    """
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


# ---------------------------------------------------------------------------
# FAISS index construction and persistence
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP (inner-product = cosine after L2-norm).

    Parameters
    ----------
    embeddings:
        Shape (n, dim), float32, L2-normalised.

    Returns
    -------
    faiss.IndexFlatIP
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[retrieval] FAISS index built: {index.ntotal} vectors, dim={dim}.")
    return index


def save_index(index: faiss.IndexFlatIP, path: str | Path) -> None:
    """Persist a FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"[retrieval] Index saved to '{path}'.")


def load_index(path: str | Path) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at '{path}'.")
    index = faiss.read_index(str(path))
    print(f"[retrieval] Loaded index from '{path}' ({index.ntotal} vectors).")
    return index


# ---------------------------------------------------------------------------
# Retrieval interface
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Encapsulates a FAISS index together with its passage corpus.

    Parameters
    ----------
    passages:
        List of passage dicts (must have ``"text"`` key).
    index:
        Corresponding FAISS index (same ordering as *passages*).
    """

    def __init__(self, passages: list[dict], index: faiss.IndexFlatIP):
        self.passages = passages
        self.index = index

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(cls, passages: list[dict]) -> "DenseRetriever":
        """Build a retriever by encoding *passages* from scratch."""
        embeddings = encode_passages(passages)
        index = build_faiss_index(embeddings)
        return cls(passages, index)

    @classmethod
    def from_disk(cls, index_path: str | Path, passages: list[dict]) -> "DenseRetriever":
        """Load a pre-built FAISS index from disk."""
        index = load_index(index_path)
        return cls(passages, index)

    # ------------------------------------------------------------------
    # Core retrieval method
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10, expand_synonyms: bool = False) -> list[dict]:
        """Encode *query* and return the *top_k* most similar passages.

        Parameters
        ----------
        query:
            Free-text symptom description.
        top_k:
            Number of results to return.
        expand_synonyms:
            If True, run the query through ``synonym_expansion.expand_query``
            before encoding. This translates snake_case Kaggle tokens (e.g.
            ``muscle_pain``) into their clinical equivalents (``myalgia``)
            so the encoder has a better chance of matching MedQuAD prose.

        Returns
        -------
        list[dict]
            Each entry contains:
            ``{"rank": int, "score": float, "text": str, "disease": str|None}``
        """
        top_k = min(top_k, self.index.ntotal)
        if expand_synonyms:
            try:
                from src.synonym_expansion import expand_query_string
                query = expand_query_string(query)
            except Exception as exc:  # pragma: no cover — optional dep
                print(f"[retrieval] synonym expansion skipped: {exc}")
        q_vec = encode_query(query)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            passage = self.passages[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "text": passage["text"],
                "disease": passage.get("disease", None),
            })
        return results

    def save(self, index_path: str | Path) -> None:
        save_index(self.index, index_path)


# ---------------------------------------------------------------------------
# Public convenience wrapper (module-level)
# ---------------------------------------------------------------------------

# Global singleton — populated by initialise() or the __main__ block.
_RETRIEVER: DenseRetriever | None = None


def initialise(
    corpus_path: str | Path | None = None,
    index_path: str | Path | None = None,
    save_index_to: str | Path | None = None,
) -> DenseRetriever:
    """Build (or load) the global retriever and return it.

    Parameters
    ----------
    corpus_path:
        JSONL file of passages.  Uses demo corpus if ``None`` / missing.
    index_path:
        If provided and exists on disk, skip encoding and load the index.
    save_index_to:
        If provided, save the newly built index here for future reuse.
    """
    global _RETRIEVER
    passages = load_corpus(corpus_path)

    if index_path and Path(index_path).exists():
        _RETRIEVER = DenseRetriever.from_disk(index_path, passages)
    else:
        _RETRIEVER = DenseRetriever.from_corpus(passages)
        if save_index_to:
            _RETRIEVER.save(save_index_to)

    return _RETRIEVER


def retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Module-level retrieve; calls initialise() with demo corpus if needed."""
    global _RETRIEVER
    if _RETRIEVER is None:
        initialise()
    return _RETRIEVER.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Dense retrieval demo — retrieves medical passages for a symptom query."
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to a JSONL corpus file (each line: {\"text\": \"...\", ...}). "
             "Falls back to built-in demo corpus if omitted or not found.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Path to a pre-built FAISS index to load (optional).",
    )
    parser.add_argument(
        "--save_index",
        default=None,
        help="Where to save the FAISS index after building (optional).",
    )
    parser.add_argument(
        "--query",
        default="fever headache chills muscle pain",
        help="Symptom query string.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of passages to retrieve.",
    )
    parser.add_argument(
        "--expand_synonyms",
        action="store_true",
        help="Expand the query with clinical synonyms before encoding "
             "(bridges Kaggle snake_case tokens to MedQuAD clinical prose).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Ensure project root is on the path so sibling modules are importable.
    sys.path.insert(0, str(Path(__file__).parent.parent))

    args = _parse_args()

    print("=" * 70)
    print("Dense Retrieval Demo")
    print("=" * 70)

    retriever = initialise(
        corpus_path=args.corpus,
        index_path=args.index,
        save_index_to=args.save_index,
    )

    print(f"\nQuery: \"{args.query}\"")
    print(f"Retrieving top-{args.top_k} passages …\n")

    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        expand_synonyms=args.expand_synonyms,
    )

    print(f"{'Rank':<5} {'Score':<8} {'Disease':<30} {'Passage (truncated)'}")
    print("-" * 90)
    for r in results:
        disease = r["disease"] or "—"
        snippet = r["text"][:60] + ("…" if len(r["text"]) > 60 else "")
        print(f"{r['rank']:<5} {r['score']:<8.4f} {disease:<30} {snippet}")

    print("\nDone.")
