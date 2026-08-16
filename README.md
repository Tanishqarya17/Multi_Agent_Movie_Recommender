# 🎬 Multi-Agent Movie Discovery with Verifiable Reasoning

> A four-agent reasoning system — **Plan → Retrieve → Verify → Explain** — that recommends films grounded in real plot data and exposes its complete reasoning process. The language model never originates a recommendation; every film traces back to retrieval over real data, and every explanation is grounded in the film's actual plot. **92.9% citation faithfulness**, **0.983 tool-use F1**, **100% well-formed agent trajectories** — together with a candid analysis of where the system underperforms a single-LLM baseline, and why that outcome is the most instructive result in the project.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1f6feb.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-free_tier-8e75ff.svg)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-semantic_search-9cf.svg)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-app-ff7c00.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**🔗 Live Demo:** [View the deployed system on Hugging Face Spaces →](https://huggingface.co/spaces/Tanishq71/multi-agent-movie-recommender)

> **Replace before publishing:** update the Space link above and the clone URL below to your exact repository and Space slugs.

---

## Overview

This project is not a movie recommender in the conventional sense — it is a demonstration of **verifiable agentic reasoning**, using movie discovery as the test domain. Four language-model-driven agents coordinate over collaborative-filtering and semantic-search tools to produce recommendations that (a) honor the user's stated constraints deterministically, (b) explain themselves grounded in real plot overviews, and (c) expose their entire reasoning trace for inspection.

| Evaluation pillar | Result |
|--------|--------|
| **Citation faithfulness** (independent LLM judge) | **0.929** mean · 74% perfectly grounded · 0 below 0.5 |
| **Tool-use** (precision / recall / F1) | 0.967 / 1.000 / **0.983** |
| **Trajectory well-formedness** | **100%** (30/30) · 3 queries self-corrected via the critic loop |
| **Win-rate vs. naive single-LLM baseline** | 16.7% → 30% (the system underperforms — the analysis below is the point) |
| **Failure modes catalogued** | 7, each with evidence and severity |

The system was built end-to-end on **free infrastructure** — Google Colab, free-tier Gemini, and Hugging Face Spaces — with no paid compute, including a model-rotation scheme to operate within free-tier rate limits.

---

## ✨ Why This Project Is Different

Most public "LLM recommender" projects prompt a model with *"what should I watch?"* and trust the answer. This project is deliberately the opposite, in three respects.

**1. Verifiable, not assumed — and measured.** The language model serves as the reasoning component of each agent, but it is never the source of a recommendation. Every film is produced by retrieval over real data (collaborative filtering and semantic search); the model only plans the search, verifies thematic fit, and writes explanations grounded in each film's real plot overview. This separation is the central thesis, and it is measured rather than asserted: **92.9% mean citation faithfulness** against an independent judge model, enforced by a defensive filter that makes it structurally impossible for the explainer to recommend a film that was not actually retrieved.

**2. Hard constraints are deterministic; only soft judgment is delegated to the model.** When the user asks for something "under 2 hours," a deterministic tool enforces the constraint — not the model, which is unreliable at exact checks. The critic agent judges only soft, thematic fit. When the system genuinely cannot satisfy the stated constraints, it relaxes them under a strict, disclosed policy and reports exactly what was loosened, rather than silently violating the request. For an impossible combination, it returns an honest no-match response instead of a fabricated one.

**3. The evaluation is rigorously honest.** Five measured pillars, a frozen 30-query test set, an independent judge, and blind position-randomized comparisons — including a headline result in which the system underperforms a naive baseline, examined in full rather than omitted. That analysis proved to be the most valuable finding in the project: a concrete demonstration of why LLM-as-judge win-rate is an inappropriate lens for a verifiable recommender.

---

## 🏗️ Architecture

Four agents are orchestrated as a **LangGraph state machine** with a bounded self-correction loop. The critic may reject a candidate pool and return control to the retriever for another pass; the loop is bounded, and every retry changes the retrieval, so termination is guaranteed.

```mermaid
graph LR
    Q([User query]) --> P[🧭 Planner]
    P -->|structured plan| R[🔎 Retriever]
    R -->|candidate pool| C[⚖️ Critic]
    C -.->|retry: relax a constraint| R
    C -->|verified pool| E[✍️ Explainer]
    E --> O([Grounded recommendations<br/>+ reasoning trace])
```

| Agent | Role | Uses LLM? |
|-------|------|-----------|
| 🧭 **Planner** | Converts a natural-language request into a validated, structured plan (anchors, mood, a rewritten semantic query, retrieval strategy, popularity preference, and hard constraints) | Yes — structured output |
| 🔎 **Retriever** | Executes the plan by orchestrating the tools: resolves anchor titles to IDs, fuses collaborative-filtering and semantic candidates, enforces hard constraints, enriches with metadata | No — deterministic |
| ⚖️ **Critic** | Judges whether candidates genuinely match the desired mood and theme; on failure, relaxes one constraint and re-retrieves under a bounded loop | Yes — batched judgment |
| ✍️ **Explainer** | Selects and ranks the best matches and writes explanations grounded strictly in each film's real plot overview | Yes — grounded generation |

### Retrieval and filtering tools

| Tool | Function | Backing |
|------|----------|---------|
| **A — Collaborative filtering** | "Users who liked *X* also liked…" via anchor fold-in | ALS (`implicit`) |
| **B — Semantic search** | Retrieves films by plot, mood, and theme | Qwen3-Embedding-0.6B + FAISS (`IndexFlatIP`) |
| **C — Metadata lookup** | Offline title, year, runtime, genres, cast, and overview | Catalog parquet |
| **D — Constrained filter** | Runtime, genre, year, and exclusions, enforced deterministically | Catalog + MovieLens genres |
| **Popularity re-ranker** | Optional and agent-invokable; blends semantic score with a popularity prior. Disabled by default and enabled only when the plan requests well-known films | Rating counts |

A design decision worth noting: **the retriever makes no LLM call.** Tool-selection reasoning resides in the planner and verification resides in the critic, so retrieval is a fast, reliable, deterministic executor of the plan — a deliberate reliability choice.

---

## ⚙️ How It Works

### The plan (structured intent)

The planner emits a schema-validated object, so every downstream agent operates on typed data rather than re-parsing prose. For *"something like Inception but less confusing, under 2 hours, nothing too obscure"* it produces approximately:

```json
{
  "anchor_titles": ["Inception"],
  "mood": "accessible mind-bending sci-fi",
  "semantic_query": "layered-reality sci-fi thriller, clear plot",
  "strategy": "hybrid",
  "prefer_popular": true,
  "genres_include": ["Sci-Fi", "Action"],
  "runtime_max": 120
}
```

Genres are constrained to the 20 MovieLens labels (and post-filtered in code); "nothing too obscure" is correctly interpreted as `prefer_popular: true`, which subsequently enables the popularity re-ranker.

### The critic loop and constraint relaxation

Hard constraints (runtime, genre, era, exclusions) are enforced deterministically before the critic evaluates any candidate, so the critic assesses only thematic fit. If too few candidates pass, the critic relaxes one constraint and re-retrieves, under a strict and disclosed policy:

- **Inviolable — never relaxed:** genre, and any titles the user explicitly excluded.
- **Relaxable, graduated, and alternating:** runtime (+20 minutes per step, up to +60) and year window (±5 per step, up to ±20).
- **Bounded:** the loop terminates when the relaxable constraints are exhausted.
- **Disclosed:** if a constraint was relaxed, the response states what was loosened and by how much. If the system had to relax to its caps and still barely matched, it reports that no films truly matched the full request and presents the closest results found only after loosening. If nothing matches at all, it returns an explicit no-match message rather than incorrect results.

This mechanism underpins the project's central claim: the system respects the stated constraints, and when it cannot, it is transparent about the compromise.

---

## 📊 Dataset

The catalog was assembled from two public sources, combined and cleaned rather than using a pre-aggregated version:

| Source | Provides | Scale |
|--------|----------|-------|
| **MovieLens 32M** | Ratings (collaborative signal) and canonical genres | 87,585 films · ~200K users · 32M ratings |
| **TMDB** | Plot overviews (the content and grounding signal) | Fetched per film, checkpointed |
| **Combined catalog** | The working film catalog | **86,262** films; **~85,300** with plot overviews long enough to embed |

Overviews were fetched from the TMDB API (checkpointed and resumable), quality-filtered, and only those with substantive text (≥10 words) were embedded into the FAISS index, so semantic search never matches on an empty or trivial plot. Collaborative filtering (ALS) is trained on the ratings; semantic search operates over the embedded overviews; and the two signals are fused at retrieval time.

---

## 🔬 Evaluation

All results below are measured on a **frozen set of 30 axis-tagged queries** spanning strategy (cf / semantic / hybrid), popularity (popular / hidden-gem / unspecified), constraint density (none → heavy → impossible), and edge cases (exclusions, negations, and vague mood-only requests). The full system was run over these queries once; every metric reads from that single canonical output, so results are reproducible and comparable. The judge is an **independent Gemini model** distinct from the one running the system, so no model grades its own output.

### 1. Citation faithfulness — 0.929 (primary strength)

A claim-level metric: each explanation is decomposed into atomic factual claims about the film, and each claim is checked against the film's real overview. Faithfulness equals supported claims divided by total claims.

- **Mean 0.929**, median **1.000**, **74%** of recommendations perfectly grounded, and **0** below 0.5 (no egregious hallucination).
- Manual review of every low-scoring case found the misses to be almost entirely **measurement artifacts** — the judge scored against a truncated overview, so true plot claims outside the snippet were marked "unsupported" — **not fabrications**. The reported 0.929 is therefore a **conservative floor**; true faithfulness is higher.

### 2. Tool-use precision / recall — 0.983 F1

Computed deterministically: the expected tool set for each query is derived from its plan, the actual tools used are extracted from the trajectory log, and the two are compared. **Precision 0.967, recall 1.000, F1 0.983** — the retriever invokes the tools the plan warrants and essentially nothing it does not.

### 3. Trajectory well-formedness — 100%

Every one of the 30 trajectories is structurally sound: the planner is present, the explainer is present, and each retriever pass is matched by one critic pass. **3 queries triggered the self-correction loop** (the critic relaxing constraints and re-retrieving), and all terminated correctly, including near-impossible queries that ran the relaxation ladder to its caps.

### 4. Win-rate against a naive baseline — an instructive negative result

The system was compared head-to-head against a **naive baseline**: a single LLM call, using the same model, with no tools and no grounding. An independent judge selected the better set, blind, with the two sets' positions randomized per query (position bias was checked and confirmed absent).

**The system underperformed the baseline.** That outcome, examined closely, is the most valuable finding in the project:

| Judging criterion | System win-rate |
|---|---|
| Naive "which is better overall" | 16.7% |
| With relaxation context provided to the judge | ~20% |
| Fame-neutral ("ignore how well-known the films are; judge only fit") | ~30% |

A review of every loss reason established the cause:

1. **A single strong LLM is an exceptionally hard baseline for common queries** — it has effectively memorized the canonical answer ("epic space operas" → Star Wars, Dune), whereas a retrieval system over a mid-size catalog returns genuinely on-theme but less-familiar films.
2. **LLM judges conflate familiarity with quality** — and instructing the judge to disregard fame only partially corrects it (win-rate rose from 16.7% to 30% but never reversed). The true thematic win-rate is higher than any of these figures, but cannot be cleanly measured with an LLM judge — itself a finding about the limits of LLM-as-judge for recommender evaluation.
3. **Win-rate measures which list appears better, which is orthogonal to this system's actual value** — grounding, deterministic constraint enforcement, transparent relaxation, verifiability, and discovery beyond the obvious. A preference judge cannot observe any of these. The baseline can neither explain why it selected a film nor guarantee that it honored a constraint; this system can, and does, measurably.

Reporting the result honestly, and explaining precisely why the metric is ill-suited to the system, is a stronger outcome than an uninformative win.

### 5. Failure-mode analysis — 7 catalogued

See the section below.

---

## 🔧 Failure Analysis

Seven named failure modes, each with evidence from the frozen runs and an assigned severity:

1. **Anchor fold-in returns sequels and near-duplicates.** "Toy Story but for adults" returned Toy Story sequels — collaborative filtering optimizes for similarity and therefore fails "like *X* but tonally different" requests. *(Medium.)*
2. **Relaxation drift on near-impossible queries.** An impossible combination can relax to the caps and still return results far from the original request; this is now flagged with an explicit disclosure. *(Low.)*
3. **Content-negation is not expressible.** The plan schema captures genre, runtime, and year but has no field for content exclusions ("no aliens," "not romantic comedies"), so that portion of the request is silently dropped. *(Medium — a genuine schema limitation.)*
4. **Costly ladder on impossible queries.** Near-impossible requests run the full relaxation ladder (up to 8 critic passes) — correct behavior, but the most computationally expensive path. *(Low.)*
5. **Retrieval underperforms parametric recall on canonical requests.** The win-rate finding: for well-known thematic queries, a single LLM's memorized answer often fits better than retrieval from an 86K catalog. *(High for win-rate; orthogonal to the system's actual value.)*
6. **Non-determinism at temperature 0.** Even pinned at temperature 0, the model can produce slightly different plans for the same query, so trajectories are not perfectly reproducible. *(Low — affects reproducibility, not correctness.)*
7. **Faithfulness misses are measurement artifacts, not fabrications.** Low faithfulness scores arose from truncated judge context and small denominators, implying true faithfulness exceeds the measured 0.929. *(Very low.)*

---

## ⚠️ Limitations

1. **Catalog scale.** Roughly 86K films with overviews — substantial, but a single strong LLM has been exposed to far more, which is precisely why it prevails on familiarity. Retrieval trades breadth of memory for grounding and verifiability.
2. **Content-negation unsupported.** "No aliens" or "not a romantic comedy" is captured only at the genre level; finer content exclusions are not representable in the plan schema.
3. **Anchor similarity is not tonal contrast.** Collaborative filtering returns behaviorally similar films, so "like *X* but different" is a known weakness.
4. **Free-tier rate limits shape the evaluation.** Metrics are computed on a 30-query set (not thousands), paced and checkpointed to operate within free-tier daily quotas via model rotation. The methodology scales; the sample size reflects the budget.
5. **Not production-validated.** This is a portfolio and research demonstration, not a deployed product.

---

## 🚀 Planned Extensions (v2)

- **A content-negation field** in the plan schema (with a corresponding post-filter) so "no aliens" is honored rather than dropped.
- **A contrast-aware anchor mode** ("like *X* but different") that deliberately searches away from an anchor's neighbors rather than toward them.
- **Re-run faithfulness with full, untruncated overviews** — manual review indicates the measured 0.929 understates true faithfulness.
- **A learned segment router** that dispatches each query to its best strategy rather than relying on planner heuristics.
- **A grounded, human-preference evaluation** (not LLM-judged) to measure the system's value where win-rate cannot.
- **Langfuse tracing** for automatic, visual observability of every run.

---

## 📁 Repository Structure

```
multi-agent-movie-discovery/
├── README.md                      Project overview (this file)
├── LICENSE                        MIT license
├── requirements.txt               Dependencies
├── notebooks/                     One notebook per build stage
│   ├── magd_tool_a_als.ipynb          ALS collaborative filtering
│   ├── magd_tool_b_semantic.ipynb     Qwen3 + FAISS semantic search + re-ranker
│   ├── magd_tools_cd.ipynb            Metadata lookup + constrained filter
│   ├── magd_llm_setup.ipynb           Gemini setup + structured output
│   ├── magd_langgraph_intro.ipynb     LangGraph fundamentals
│   ├── magd_agents.ipynb              The four agents + full graph
│   ├── magd_eval.ipynb                The five-pillar evaluation
│   └── magd_app.ipynb                 The Gradio application
├── src/
│   ├── tools/                     tool_a_als, tool_b_semantic, tool_c_details, tool_d_filter, reranker
│   ├── agents/                    planner, retriever, critic, explainer
│   ├── graph/                     state (AgentState), build (build_graph)
│   └── llm/                       gemini_client, gemini_rotating
├── eval/
│   ├── datasets/                  test_queries.json (30 frozen, axis-tagged)
│   └── results/                   system_runs, faithfulness_scores, tooluse_trajectory, winrate_*, failure_modes
├── figures/                       architecture and application screenshots
└── deploy/                        The shipped application (app.py, requirements.txt, README, artifacts)
```

---

## 🔁 Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/Tanishqarya17/multi-agent-movie-discovery.git
cd multi-agent-movie-discovery
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project was developed entirely on **free Google Colab** (Python 3.12). A GPU accelerates semantic embedding but is not required; all components run on CPU.

### 3. Acquire the data

Download **MovieLens 32M** from [GroupLens](https://grouplens.org/datasets/movielens/), and fetch plot overviews from the [TMDB API](https://developer.themoviedb.org/) (a free API key is required; the fetch notebook is checkpointed and resumable). Raw data is not redistributed in this repository.

### 4. Provide an LLM key

Set a free **Google AI Studio** (Gemini) API key as `GOOGLE_API_KEY`. The system pins one Flash-Lite model and rotates across others to remain within free-tier daily quotas.

### 5. Run the notebooks in order

`magd_tool_a_als` → `magd_tool_b_semantic` → `magd_tools_cd` → `magd_llm_setup` → `magd_langgraph_intro` → `magd_agents` → `magd_eval` → `magd_app`. Each stage is self-documenting and verifies its outputs before the next.

### 6. Launch the application locally

```bash
cd deploy
pip install -r requirements.txt
python app.py
```

This launches the same Gradio interface that is deployed live, reading the artifacts alongside `app.py`.

---

## 🛠️ Tech Stack

`Python` · `LangGraph` · `Google Gemini` (`google-genai`) · `Pydantic` · `Qwen3-Embedding-0.6B` · `FAISS` · `implicit` (ALS) · `sentence-transformers` · `pandas` · `Gradio` · `Hugging Face Spaces` — built entirely on free Google Colab.

---

## 🙏 Acknowledgments

- **GroupLens (MovieLens 32M)** — ratings and canonical genres.
- **The Movie Database (TMDB)** — plot overviews. This product uses the TMDB API but is not endorsed or certified by TMDB.
- **Qwen team** — the Qwen3 embedding model.
- **LangChain / LangGraph** and **`implicit`** — the orchestration and retrieval building blocks.

---

## 📫 Contact

**Tanishq Arya** — B.Tech, AI & Data Science

- GitHub: [@Tanishqarya17](https://github.com/Tanishqarya17)
- Hugging Face: [@Tanishq71](https://huggingface.co/Tanishq71)
- Email: [tanishqarya789@gmail.com](mailto:tanishqarya789@gmail.com)
- LinkedIn: [@TanishqArya](https://www.linkedin.com/in/tanishq-arya-b10598292/)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
