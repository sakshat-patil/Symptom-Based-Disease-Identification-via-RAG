# Contributors

CMPE 255 Spring 2026, San Jose State University

## Team

| Name | SJSU ID | Role |
| --- | --- | --- |
| Sakshat Nandkumar Patil | 018318287 | Data ETL (Kaggle + Synthea), FP-Growth mining, mining-side scoring, evaluation harness, latency benchmarking |
| Vineet Kumar | 019140433 | MedQuAD preprocessing, embedding backends (MiniLM + PubMedBERT + Azure OpenAI), FAISS and Pinecone vector stores, dense retriever, cross-encoder rerank, FastAPI inference service, OpenAI structured explainer |
| Aishwarya Madhave | 019129110 | Synonym dictionary, fusion reranker, alpha sweep, ablation harness, Next.js web UI components, plots, paper writing lead |

All three authors are enrolled in CMPE 255 Spring 2026. No external collaborators contributed to this project.

## How we worked

We met twice a week (Tue + Sat). Sakshat owned the data side end to end, Vineet owned everything in the retrieval stack, and Aishwarya owned the fusion logic and the demo. The report and the IEEE-format slide deck were written collaboratively; the final reproducible PDFs and the slide deck are generated from scripts in `code/scripts/` so anyone can rebuild them.

When components needed to talk to each other (e.g., the cross-encoder needs the bi-encoder's top-K), we paired live for an hour to nail the interface and only then split the implementation.

## Decision log (the things we argued about)

- **Single-source vs dual-source training data.** We carry both a curated Kaggle-style transaction table and a working Synthea ETL. Synthea was tempting because it produces realistic FHIR bundles, but per-row label noise hurt the rule-mining ablation. We kept Synthea functional for completeness and run the headline experiments on the cleaner table. Sakshat documented this in `synthea_etl.py`.

- **Bi-encoder choice.** MiniLM is small and fast, PubMedBERT is biomedical. Vineet ran both and we shipped both. PubMedBERT wins on raw retrieval-only Recall@10, but in our fused setup the encoder difference is small relative to the synonym-expansion lift, which is why the headline number uses MiniLM+syn.

- **Cross-encoder always-on?** We toyed with putting the cross-encoder on the default path. It improves ordering at K below 5 but adds about 100ms per query. Aishwarya argued against making it default for the demo, so we left it as a sidebar toggle.

- **LLM explainer.** The proposal calls for a generative LLM. We initially shipped a local FLAN-T5 path so the demo would work fully offline. Once we wired up the Azure OpenAI integration for the production demo, the FLAN-T5 path stopped paying for itself (slower, lower quality, an extra dependency). The system now ships two explainers: a deterministic citation-faithful template (always works, no key) and an OpenAI-backed structured-JSON explainer (used in the live demo against `truestar-gpt-5.3-chat`). Both keep every claim cited back to a passage id, and the OpenAI path auto-falls-back to the template on any error.

- **min_support = 0.005 vs 0.01.** Check-in 3 used 0.01 and left 21 of 41 diseases without rules. We dropped to 0.005 in Check-in 4 (and kept it for the final), which gives us 100% disease coverage at the cost of a noisier rule set. The overlap weighting in `MiningScorer.score` neutralises most of the noise.

- **alpha default.** The alpha sweep is flat across 0.1 to 0.4. We picked 0.3 to match the operating point we publicly committed to in Check-in 4.
