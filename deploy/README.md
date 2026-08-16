---
title: Multi-Agent Movie Recommender System
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent film recommender with reasoning
python: "3.12.0"
---

# 🎬 Multi-Agent Movie Recommender System

A multi-agent reasoning system that recommends films grounded in real plot data and exposes its complete reasoning process. Four agents — **Planner → Retriever → Critic → Explainer** — coordinate over collaborative-filtering and semantic-search tools to produce recommendations that honor stated constraints and explain themselves. The full agent reasoning trace is available on request under "🧠 See how these were chosen."

**Citation faithfulness:** 0.929 (independent judge) · **Tool-use F1:** 0.983 · **Well-formed trajectories:** 100%

## What it does

- Interprets natural-language requests, including implicit constraints — for example, *"something like Inception but less confusing, under 2 hours, nothing too mainstream."*
- Retrieves candidates through ALS collaborative filtering and Qwen3 semantic search over approximately 86,000 films.
- Enforces hard constraints (runtime, genre, era, and exclusions) deterministically, and when they cannot be satisfied, reports what was relaxed rather than silently violating the request.
- Explains each recommendation grounded in the film's actual plot, without invented details.
- Exposes the complete four-agent reasoning trace for inspection.

## ⚠️ Note

This is a portfolio and research demonstration, not a production system. Recommendations are drawn from the MovieLens and TMDB catalog. The first request after the Space wakes may take one to two minutes while the semantic model loads on boot. Constraint relaxation, when it occurs, is always disclosed.

## ✨ Why this project is different

Most language-model recommenders prompt a model directly and trust the answer. This system never permits the model to originate a recommendation: every film is produced by retrieval over real data, and the model only plans the search, verifies thematic fit, and explains. That grounding is measured rather than assumed — 92.9% mean citation faithfulness against an independent judge model. Hard constraints are enforced deterministically rather than delegated to the model, and the system honestly reports when no match exists rather than violating the request.

## 🛠️ Built with

- **LangGraph** — four-agent orchestration with a bounded self-correction loop
- **Google Gemini** (free tier) — the reasoning component of each agent
- **Qwen3-Embedding + FAISS** — semantic plot search
- **implicit (ALS)** — collaborative filtering
- **Gradio** — this interface

## Author

**Tanishq Arya** — B.Tech, AI & Data Science

The complete project, architecture, and five-pillar evaluation — including a candid analysis of where the system underperforms a naive baseline — are available on the [GitHub repository](https://github.com/Tanishqarya17/multi-agent-movie-discovery).
