"""One-shot script: push embeddings into a Pinecone index.

Two flows depending on which backend you pick:

    --backend pubmedbert   reads cached vectors from
                              data/processed/embeddings_pubmedbert.npy
                              (already produced by `python -m src.retrieval`)
    --backend azure-openai encodes the 24,063 MedQuAD passages live via
                              the Azure OpenAI endpoint configured by
                              OPENAI_API_KEY + OPENAI_BASE_URL +
                              OPENAI_EMBEDDING_MODEL. Costs roughly $0.80
                              and takes ~3 minutes.

Both flows upsert metadata fields (source, focus, question, text excerpt)
so the Pinecone Query API can metadata-filter without us round-tripping
back through Python.

Usage:
    PINECONE_API_KEY=...  PINECONE_INDEX_NAME=255-data-mining \\
    OPENAI_API_KEY=...    OPENAI_BASE_URL=...                 \\
    python scripts/seed_pinecone.py --backend azure-openai
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_backends import get_backend  # noqa: E402
from src.retrieval import load_passages  # noqa: E402
from src.vector_store import PineconeStore  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def _load_or_compute_embeddings(backend_name: str, passages: list[dict]) -> np.ndarray:
    """For local backends, load the cached .npy. For Azure, encode now."""
    if backend_name in ("minilm", "pubmedbert"):
        emb_path = PROCESSED / f"embeddings_{backend_name}.npy"
        if not emb_path.exists():
            sys.exit(f"missing {emb_path}; run "
                      f"`python -m src.retrieval --backend {backend_name}` first")
        return np.load(emb_path)

    if backend_name in ("azure-openai", "openai", "azure"):
        backend = get_backend(backend_name)
        print(f"[pinecone] encoding {len(passages)} passages via "
              f"{backend.name} ({backend.model}, dim={backend.dim}); "
              f"this is the slow step")
        texts = [p["text"] for p in passages]
        # Azure has a token-per-request cap; the backend already batches
        # at 16 inputs. We just call it once.
        emb = backend.encode(texts, batch_size=16, show_progress=True)
        # Cache locally so the user doesn't pay twice.
        cache = PROCESSED / "embeddings_azure-openai.npy"
        np.save(cache, emb)
        print(f"[pinecone] cached {cache}")
        return emb

    raise ValueError(f"unknown backend for seeding: {backend_name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="pubmedbert",
                    choices=["minilm", "pubmedbert", "azure-openai"])
    args = p.parse_args()

    passages = load_passages()
    emb = _load_or_compute_embeddings(args.backend, passages)
    if len(passages) != emb.shape[0]:
        sys.exit(f"passage/embedding count mismatch: {len(passages)} vs "
                  f"{emb.shape[0]}")

    print(f"[pinecone] backend={args.backend} dim={emb.shape[1]} "
          f"passages={len(passages)} index={os.environ.get('PINECONE_INDEX_NAME','medical-rag-medquad')}")
    store = PineconeStore(dim=emb.shape[1])
    store.upsert(passages, emb)
    print(f"[pinecone] done. index={store.index_name}")


if __name__ == "__main__":
    main()
