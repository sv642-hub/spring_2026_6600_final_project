"""
ModelLens Gradio shell — optimized for live class demos.

Run:  python -m app.main
   or:  gradio app/main.py (if configured)
"""

from __future__ import annotations

import gradio as gr

from app.components import (
    build_overview,
    load_huggingface_lens,
    presentation_story,
    run_attn_fig,
    run_embed_fig,
    run_logit_figs,
    run_patch_fig,
    run_residual_fig,
)
from app.demo_data import (
    APP_TITLE,
    DEFAULT_CORRUPTED,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
)

_GRADIO_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)
_GRADIO_CSS = """
.gr-markdown h1 { font-weight: 700; letter-spacing: -0.02em; }
footer { visibility: hidden; }
"""


def _need_lens(lens):
    if lens is None:
        raise gr.Error("Load a Hugging Face model first (Overview tab).")
    return lens


def create_app():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Guided tour of **attention**, **logit lens**, **activation patching**, "
            "and **residual / embedding** views. Built on Plotly for crisp, hover-rich figures."
        )

        lens_state = gr.State(None)

        with gr.Row():
            model_in = gr.Dropdown(
                choices=["gpt2", "gpt2-medium", "distilgpt2"],
                value=DEFAULT_MODEL,
                label="Hugging Face causal LM",
            )
            load_btn = gr.Button("Load model", variant="primary")
            load_status = gr.Markdown()

        def _load(name):
            lens, _ = load_huggingface_lens(name)
            return lens, f"Loaded **`{name}`** — tokenizer attached; use eager attention for weights."

        load_btn.click(_load, inputs=[model_in], outputs=[lens_state, load_status])

        with gr.Tabs():
            with gr.Tab("1 · Model overview"):
                prompt_ov = gr.Textbox(
                    label="Prompt for shape trace",
                    value=DEFAULT_PROMPT,
                    lines=2,
                )
                run_ov = gr.Button("Refresh overview", variant="primary")
                summary_md = gr.Markdown()
                fig_shape = gr.Plot(label="Shape trace")
                mermaid_md = gr.Markdown()

                def _ov(prompt, lens):
                    lens = _need_lens(lens)
                    fig, md, mer = build_overview(lens, prompt)
                    return md, fig, mer

                run_ov.click(_ov, [prompt_ov, lens_state], [summary_md, fig_shape, mermaid_md])

            with gr.Tab("2 · Attention"):
                prompt_a = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2)
                layer_i = gr.Slider(0, 11, value=0, step=1, label="Layer index")
                head_i = gr.Slider(0, 11, value=0, step=1, label="Head index")
                run_a = gr.Button("Plot attention", variant="primary")
                fig_a = gr.Plot()

                def _attn(p, li, hi, lens):
                    lens = _need_lens(lens)
                    return run_attn_fig(lens, p, int(li), int(hi))

                run_a.click(_attn, [prompt_a, layer_i, head_i, lens_state], [fig_a])

            with gr.Tab("3 · Logit lens"):
                prompt_l = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2)
                run_l = gr.Button("Run logit lens", variant="primary")
                fig_le = gr.Plot(label="Top-1 probability trajectory")
                fig_lh = gr.Plot(label="Top-k heatmap across layers")

                def _logit(p, lens):
                    lens = _need_lens(lens)
                    evo, heat = run_logit_figs(lens, p)
                    return evo, heat

                run_l.click(_logit, [prompt_l, lens_state], [fig_le, fig_lh])

            with gr.Tab("4 · Activation patching"):
                clean = gr.Textbox(label="Clean prompt", value=DEFAULT_PROMPT, lines=2)
                corrupted = gr.Textbox(
                    label="Corrupted prompt (same length)",
                    value=DEFAULT_CORRUPTED,
                    lines=2,
                )
                run_p = gr.Button("Run patching", variant="primary")
                patch_html = gr.HTML()
                fig_p = gr.Plot()

                def _patch(c, r, lens):
                    lens = _need_lens(lens)
                    fig, html = run_patch_fig(lens, c, r)
                    return html, fig

                run_p.click(_patch, [clean, corrupted, lens_state], [patch_html, fig_p])

            with gr.Tab("5 · Residual & embeddings"):
                prompt_re = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2)
                run_re = gr.Button("Residual stream", variant="primary")
                run_em = gr.Button("Embedding similarity")
                fig_re = gr.Plot(label="Residual contribution")
                fig_em = gr.Plot(label="Cosine similarity")

                def _res(p, lens):
                    lens = _need_lens(lens)
                    return run_residual_fig(lens, p)

                def _emb(p, lens):
                    lens = _need_lens(lens)
                    return run_embed_fig(lens, p)

                run_re.click(_res, [prompt_re, lens_state], [fig_re])
                run_em.click(_emb, [prompt_re, lens_state], [fig_em])

            with gr.Tab("6 · Presentation story"):
                gr.Markdown(
                    "**One-click narrative** — structure → attention → logit lens → patching. "
                    "Use the same prompts as tab 4 for a coherent causal story."
                )
                ps_clean = gr.Textbox(label="Clean prompt", value=DEFAULT_PROMPT, lines=2)
                ps_cor = gr.Textbox(label="Corrupted prompt", value=DEFAULT_CORRUPTED, lines=2)
                run_story = gr.Button("Run full story", variant="primary")
                story_md = gr.Markdown()
                s_shape = gr.Plot()
                s_attn = gr.Plot()
                with gr.Row():
                    s_logit_hm = gr.Plot()
                    s_logit_evo = gr.Plot()
                s_patch = gr.Plot()

                def _story(c, r, lens):
                    lens = _need_lens(lens)
                    fig_shape, fig_attn, hm, evo, fig_patch, sm = presentation_story(
                        lens, c, r
                    )
                    return sm, fig_shape, fig_attn, hm, evo, fig_patch

                run_story.click(
                    _story,
                    [ps_clean, ps_cor, lens_state],
                    [story_md, s_shape, s_attn, s_logit_hm, s_logit_evo, s_patch],
                )

        gr.Markdown(
            "_Tip: first load **`gpt2`** on a good network connection; subsequent analysis is local._"
        )

    return demo


def main():
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        theme=_GRADIO_THEME,
        css=_GRADIO_CSS,
    )


if __name__ == "__main__":
    main()
