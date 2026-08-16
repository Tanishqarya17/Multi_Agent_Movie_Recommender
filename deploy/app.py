# ---- OPENBLAS must be set before numpy loads (ALS speed on CPU) ----
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import gradio as gr
from llm.gemini_rotating import RotatingGeminiLLM
from tools.tool_a_als import ToolA
from tools.tool_b_semantic import ToolB
from tools.tool_c_details import ToolC
from tools.tool_d_filter import ToolD
from tools.reranker import PopularityReranker
from graph.build import build_graph

HERE = os.path.dirname(__file__)
ALS_DIR    = os.path.join(HERE, 'artifacts/als')
FAISS_DIR  = os.path.join(HERE, 'artifacts/faiss')
CATALOG    = os.path.join(HERE, 'data/processed/catalog/catalog.parquet')
MOVIES_CSV = os.path.join(HERE, 'data/raw/ml-32m/movies.csv')

# ---- key from environment (set as a Space secret) ----
API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set. Add it as a Space secret.")

# ---- build the system ONCE at startup ----
OVERFLOW = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
llm = RotatingGeminiLLM(API_KEY, pinned_model="gemini-3.5-flash-lite", overflow_pool=OVERFLOW)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tool_a   = ToolA(ALS_DIR, MOVIES_CSV)                        # anchor fold-in only (no user_item)
tool_b   = ToolB(FAISS_DIR, CATALOG, device=device)
tool_c   = ToolC(CATALOG, MOVIES_CSV)
tool_d   = ToolD(CATALOG, MOVIES_CSV)
reranker = PopularityReranker(CATALOG)
app_graph = build_graph(llm, tool_a, tool_b, tool_c, tool_d, reranker, CATALOG)


def run_query(query):
    if not query or not query.strip():
        yield "Please enter what you're in the mood for.", ""
        return
    initial = {"query": query, "iterations": 0, "trajectory": [],
               "relax_state": {"runtime_steps": 0, "year_steps": 0, "next": "runtime"}}
    trace_lines, final_state = [], {}
    for chunk in app_graph.stream(initial):
        for node, update in chunk.items():
            traj = update.get("trajectory", [])
            trace_lines.append("- " + (traj[-1] if traj else f"{node}: ..."))
            final_state.update(update)
            yield "⏳ *thinking...*", "\n".join(trace_lines)

    recs = final_state.get("recommendations") or []
    fb = final_state.get("critic_feedback") or ""
    header = f"> ⚠️ {fb}\n\n" if (fb.startswith("STRETCH") or fb == "NO_FULL_MATCH" or fb.startswith("Relaxed")) else ""
    if not recs:
        body = "I couldn't find movies matching all your constraints. Try loosening one (runtime, era, or genre)."
    else:
        body = "\n\n".join(
            f"**{i}. {r['title']} ({r.get('year')})**  \n_{'|'.join(r.get('genres') or [])}_  \n{r['explanation']}"
            for i, r in enumerate(recs, 1))
    yield header + body, "\n".join(trace_lines)


EXAMPLES = [
    "something like Inception and The Matrix but less confusing, under 2 hours",
    "an underrated 80s horror movie, short, nothing mainstream",
    "feel-good animated family movies for kids",
    "twisty crime thrillers with unreliable narrators",
    "a cozy film for a rainy Sunday",
]

with gr.Blocks(title="Multi-Agent Movie Recommender System", theme=gr.themes.Soft(primary_hue="violet")) as demo:
    with gr.Column():
        gr.Markdown(
            "<div style='text-align:center'>"
            "<h1>🎬 Multi-Agent Movie Recommender</h1>"
            "<p style='font-size:1.1em; color:#666; margin-top:-8px'>"
            "Tell me what you're in the mood for — I'll find films that actually fit, and show you why.</p>"
            "</div>")
        query_box = gr.Textbox(label="", placeholder="e.g. a dark sci-fi thriller under 2 hours, nothing too mainstream…",
                               lines=2, autofocus=True)
        submit = gr.Button("✨ Find my films", variant="primary", size="lg")
        gr.Examples(EXAMPLES, inputs=query_box, label="Need inspiration? Try one of these:")
        output = gr.Markdown()
        with gr.Accordion("🧠 See how these were chosen", open=False):
            gr.Markdown("_Behind every recommendation, four agents plan, search, verify, and explain. "
                        "Here's their step-by-step reasoning for your request:_")
            trace = gr.Markdown()
    submit.click(run_query, inputs=query_box, outputs=[output, trace])
    query_box.submit(run_query, inputs=query_box, outputs=[output, trace])

if __name__ == "__main__":
    demo.launch()
