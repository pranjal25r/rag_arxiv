# Citera — Evaluation-Driven RAG over Diffusion-Model Papers

A retrieval-augmented question-answering system over **~304 arXiv papers on diffusion models**, built to *measure and improve* retrieval quality rather than just ship a chatbot. Every answer is grounded in retrieved papers and cited; the system **abstains when it lacks evidence** instead of hallucinating; and retrieval strategies are **benchmarked with an LLM-as-judge harness**.

---

## Headline result

On a deliberately scaled, noisy corpus, **cross-encoder reranking improved retrieval context precision by 50%** (0.25 → 0.375) while *also* lifting answer faithfulness, with perfect abstention on out-of-domain questions maintained throughout.

| Config | Faithfulness | Answer Relevancy | Context Precision | Abstention Acc. |
|---|---:|---:|---:|---:|
| Naive dense retrieval | 0.933 | 0.967 | 0.250 | 1.00 |
| + Cross-encoder rerank | **0.950** | **0.975** | **0.375** | 1.00 |

*Scores are LLM-as-judge (0–1) over an 11-question eval set: 8 in-domain diffusion questions + 3 out-of-domain. Abstention accuracy = fraction of out-of-domain questions correctly refused.*

---

## How it works

```
arXiv API ──► abstracts ──► sentence-transformer embeddings ──► FAISS index
                                                                    │
question ──► embed ──► retrieve top-k ──►(optional cross-encoder rerank)──► grounded LLM ──► cited answer / abstain
                                                                    │
                                              LLM-as-judge ◄── (question, answer, contexts)
```

1. **Ingestion** — pull diffusion-model papers from the arXiv API across ~10 subtopic queries (DDPM, score-based models, classifier-free guidance, latent diffusion, DiT, DDIM, text-to-image, …), deduplicate, and store abstracts. Abstracts (not full PDFs) keep the corpus clean and dense so retrieval quality isn't fighting messy PDF parsing.
2. **Indexing** — embed each abstract with `all-MiniLM-L6-v2`, normalize, and store in a FAISS inner-product index (normalized vectors + inner product = cosine similarity).
3. **Retrieval** — two strategies are benchmarked:
   - *Naive dense*: take the top-4 by embedding similarity.
   - *Reranked*: cast a wider net (top-20 by embedding), then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores each (question, passage) pair and keeps the best 4.
4. **Generation** — the LLM (Llama-3.3-70B via Groq) answers using **only** the retrieved context, cites sources by number, and is instructed to refuse — "I don't have enough information to answer that" — when the context doesn't support an answer.
5. **Evaluation** — an LLM-as-judge harness scores every answer on **faithfulness** (are all claims supported by the context?), **answer relevancy** (does it address the question?), and **context precision** (what fraction of retrieved chunks are relevant?), plus **abstention accuracy** on out-of-domain questions.

---

## Key findings

**1. Reranking's value depends on retrieval noise.** On an initial small (~100-paper), tightly on-topic corpus, reranking made almost no difference (context precision 0.45 → 0.46) — dense retrieval already surfaced the few relevant papers, leaving nothing to reorder. Only after scaling the corpus to ~304 papers, which introduced realistic near-miss noise, did reranking demonstrate clear value (0.25 → 0.375). The lesson: a reranker earns its keep when the first-stage retriever makes ranking errors, which happens in larger, noisier corpora — not in toy ones.

**2. Better retrieval improves answer faithfulness.** Cleaner context from reranking didn't just raise precision — it lifted faithfulness (0.933 → 0.950), because the generator had fewer irrelevant chunks to be misled by.

**3. The system stays reliable under imperfect retrieval.** Even at a modest 0.375 context precision, faithfulness held at 0.95 and abstention at 100%. Grounding constraints plus abstention make the system robust to noisy retrieval rather than brittle.

**On absolute numbers:** context precision is capped low here because each question has only ~1–2 truly relevant papers in a ~304-paper corpus, so precision@4 can't exceed ~0.25–0.5 by construction. The *improvement* from reranking is the headline, not the absolute value.

---

## Design decisions

- **Abstracts over full text** — dense, self-contained, and fast to iterate on; isolates retrieval quality from PDF-parsing noise.
- **Cosine via normalized inner product** — simple, exact search appropriate for a few-hundred-document corpus (no approximate index needed).
- **Wide-net retrieve → rerank** — the reranker can only promote relevant papers it's given, so the first stage fetches 20 candidates before narrowing to 4.
- **Abstention as a first-class behavior** — currently enforced via grounding prompt; a confidence-based version using retrieval scores is the natural next step.
- **Custom LLM-as-judge instead of a library** — implemented faithfulness / relevancy / context-precision using their standard definitions, on the same inference stack as the system. This avoids dependency churn and, more importantly, makes the metric fully explainable end to end.

---

## What I'd do next

- **Confidence-based abstention** — trigger refusal from retrieval-score thresholds, not just prompt instructions.
- **Hybrid retrieval** — fuse BM25 (keyword) with dense embeddings; exact-term matching often helps on technical jargon.
- **Stronger / independent judge** — cross-validate the LLM-as-judge scores against the RAGAS library and a different judge model to reduce self-evaluation bias.
- **Full-text chunking** — index paper bodies, not just abstracts, with proper chunking and overlap.

---

## Tech stack

Python · FAISS · sentence-transformers (`all-MiniLM-L6-v2`) · cross-encoder (`ms-marco-MiniLM-L-6-v2`) · Groq (Llama-3.3-70B) · arXiv API

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install groq arxiv sentence-transformers faiss-cpu numpy python-dotenv
```

Create a `.env` file (and keep it out of git):

```
GROQ_API_KEY=your-key-here
```

Then run the pipeline:

```bash
python3 fetch_papers.py     # pull diffusion papers from arXiv → data/papers.json
python3 build_index.py      # embed abstracts → FAISS index
python3 ask.py "what is classifier-free guidance?"   # interactive Q&A
python3 evaluate.py         # run the naive-vs-reranked ablation
```

---

## Project structure

```
rag_arxiv/
├── fetch_papers.py        # pull diffusion papers from arXiv → data/papers.json
├── build_index.py         # embed abstracts → FAISS index
├── ask.py                 # interactive: retrieve + grounded, cited answer
├── evaluate.py            # ablation harness: naive vs reranked, LLM-as-judge
├── eval_questions.json    # in-domain + out-of-domain eval set
├── data/                  # papers.json, papers.index
├── results_ablation.json  # saved metrics
├── .env                   # GROQ_API_KEY (gitignored)
└── README.md
```
