# Arabic-Islamic-Assessment — Retrieval-Augmented Generation (RAG) Pipeline

**Short description**
This repository implements a compact, efficient Retrieval-Augmented Generation (RAG) pipeline designed to answer multiple-choice questions in Islamic knowledge (theology, jurisprudence, biography, ethics) in Arabic. It combines domain-specific embeddings, a vector-store retriever, and a lightweight Large Language Model (LLM) generator to produce grounded, high-accuracy answers.

**Paper / Reference**
The approach and experimental results are described in the ACL ArabicNLP 2025 proceedings. You can read the full paper here: [Paper Link](https://aclanthology.org/2025.arabicnlp-sharedtasks.133/)

---

## Key features

* Preprocessing for Arabic classical texts (diacritics removal, Tatweel removal, normalization, chunking)
* Embedding-based retrieval using **Muffakir_Embedding** (768-dim)
* Vector store: **Qdrant** (HNSW, cosine similarity)
* Generator: **Gemini 2.5 Flash Lite** (or any compatible LLM)
* Simple, reproducible configuration that achieved **87% accuracy** on the QIAS 2025 test set (ranked 5th / 10 teams)

---

## Data & preprocessing

* Source documents: curated collection of classical Islamic reference works (Usul al-Fiqh, Al-Itqan, Sirah, Tafsir collections, etc.).
* Cleaning steps: remove diacritics, Tatweel, emails, urls, normalise whitespace/punctuation.
* Chunking: 400-character chunks with 100-character overlap (configurable).

---

## Architecture

1. **Ingestion**: load canonical texts, clean, chunk, annotate with metadata.
2. **Embedding**: compute Muffakir_Embedding (768-dim) per chunk.
3. **Indexing**: store vectors in Qdrant using HNSW with recommended HNSW parameters.
4. **Retrieval**: embed query -> Top-K (K=10) cosine similarity search.
5. **Prompt construction**: assemble Top-K chunks with the question and choices into a structured prompt.
6. **Generation**: feed prompt to the LLM (Gemini 2.5 Flash Lite) and parse the returned answer.

---

## Reproducible configuration (final)

* Chunk size: `400` characters
* Overlap: `100` characters
* Top-K retrieval: `10`
* Embedding dim: `768` (Muffakir_Embedding)
* Vector count (example): ~15,000
* Qdrant / HNSW parameters: `m=64`, `ef_construct=1024`, `full_scan_threshold=0`

---

## Results (summary)

* **Test set accuracy:** 87.00% (official test set, QIAS 2025 Subtask 2)
* **Ranking:** 5th out of 10 teams
* Performance by difficulty: Beginner (≈89.1%), Intermediate (≈83.4%), Advanced (≈75.4%).

---

## Tips & notes

* Muffakir embeddings were a major factor in performance; try alternative domain-tuned embeddings if available.
* Reranking did not always help; direct retrieval + LLM gave more reliable results in the reported experiments.
* Use overlap and chunk-size tuning to trade off retrieval granularity vs. context size.

---

## Authors & contact

Original paper authors: Mohamed Samy, Mayar Boghdady, Marwan El Adawi, Mohamed Nassar, Ensaf Hussein.
For questions about this implementation or the paper, contact: `mohamedsamyy02@gmail.com`.

---
