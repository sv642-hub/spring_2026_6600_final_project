"""Reusable analysis + plotting helpers for the Gradio shell."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from modellens import ModelLens
from modellens.analysis.activation_patching import run_activation_patching
from modellens.analysis.attention import run_attention_analysis
from modellens.analysis.embeddings import run_embeddings_analysis
from modellens.analysis.logit_lens import run_logit_lens
from modellens.analysis.residual_stream import run_residual_analysis
from modellens.visualization.activation_patching import (
    format_patching_summary_html,
    plot_patching_importance_bar,
)
from modellens.visualization.attention import plot_attention_heatmap
from modellens.visualization.embeddings import plot_embedding_similarity_heatmap
from modellens.visualization.logit_lens import plot_logit_lens_evolution, plot_logit_lens_heatmap
from modellens.visualization.residuals import plot_residual_contributions
from modellens.visualization.shapes import (
    compute_shape_trace,
    plot_shape_trace_table,
    shape_trace_mermaid,
)


def load_huggingface_lens(model_name: str) -> Tuple[ModelLens, Any]:
    """Load tokenizer + causal LM and wrap with ModelLens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.eval()
    lens = ModelLens(model, backend="huggingface")
    lens.adapter.set_tokenizer(tokenizer)
    return lens, tokenizer


def tokenize(lens: ModelLens, text: str) -> Dict[str, torch.Tensor]:
    return lens.adapter.tokenize(text)


def transformer_block_layer_names(model: torch.nn.Module) -> List[str]:
    """GPT-2 style blocks; extend later for other causal LMs."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "n_layer"):
        return [f"transformer.h.{i}" for i in range(cfg.n_layer)]
    return []


def build_overview(
    lens: ModelLens, prompt: str
) -> Tuple[Any, str, str]:
    """Shape table figure, markdown summary, mermaid snippet."""
    tokens = tokenize(lens, prompt)
    rows = compute_shape_trace(lens, tokens)
    fig = plot_shape_trace_table(rows, max_rows=60)
    s = lens.summary()
    md = (
        f"**Backend:** `{s['backend']}`  \n"
        f"**Parameters:** {s['total_parameters']:,}  \n"
        f"**Named modules:** {len(s['layer_names'])}  \n"
        f"**Hooks attached:** {s['hooks_attached']}  \n"
    )
    mer = "```mermaid\n" + shape_trace_mermaid(rows, max_nodes=20) + "\n```"
    return fig, md, mer


def run_attn_fig(
    lens: ModelLens, prompt: str, layer_index: int, head_index: int
) -> Any:
    tokens = tokenize(lens, prompt)
    ar = run_attention_analysis(lens, tokens)
    n_layers = len(ar.get("layers_ordered") or [])
    if n_layers:
        layer_index = int(min(max(0, layer_index), n_layers - 1))
    w = next(iter(ar["attention_maps"].values()))["weights"]
    if hasattr(w, "shape") and w.ndim == 4:
        nh = int(w.shape[1])
        head_index = int(min(max(0, head_index), max(0, nh - 1)))
    return plot_attention_heatmap(
        ar, layer_index=layer_index, head_index=head_index
    )


def run_logit_figs(lens: ModelLens, prompt: str) -> Tuple[Any, Any]:
    tokens = tokenize(lens, prompt)
    lr = run_logit_lens(
        lens, tokens, tokenizer=lens.adapter._tokenizer, top_k=5
    )
    evo = plot_logit_lens_evolution(lr)
    heat = plot_logit_lens_heatmap(lr, top_ranks=5)
    return evo, heat


def run_patch_fig(
    lens: ModelLens, clean: str, corrupted: str
) -> Tuple[Any, str]:
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
    fig = plot_patching_importance_bar(pr, use_normalized=True)
    html = format_patching_summary_html(pr)
    return fig, html


def run_residual_fig(lens: ModelLens, prompt: str) -> Any:
    tokens = tokenize(lens, prompt)
    names = transformer_block_layer_names(lens.model)
    if not names:
        raise ValueError("Could not infer transformer blocks for residual analysis.")
    rr = run_residual_analysis(lens, tokens, layer_names=names)
    return plot_residual_contributions(rr, mode="relative")


def run_embed_fig(lens: ModelLens, prompt: str) -> Any:
    tokens = tokenize(lens, prompt)
    er = run_embeddings_analysis(lens, tokens)
    return plot_embedding_similarity_heatmap(er)


def presentation_story(
    lens: ModelLens, prompt: str, corrupted: str
) -> Tuple[Any, Any, Any, Any, Any, str]:
    """Curated pipeline: shape → attention → logit lens → patching + summary."""
    tokens = tokenize(lens, prompt)
    rows = compute_shape_trace(lens, tokens)
    fig_shape = plot_shape_trace_table(rows, max_rows=50, title="Story — shape trace")

    ar = run_attention_analysis(lens, tokens)
    fig_attn = plot_attention_heatmap(ar, layer_index=0, head_index=0)

    lr = run_logit_lens(
        lens, tokens, tokenizer=lens.adapter._tokenizer, top_k=5
    )
    fig_logit_hm = plot_logit_lens_heatmap(lr, top_ranks=5)
    fig_logit_evo = plot_logit_lens_evolution(lr, rank_index=0)

    clean_t = tokens
    cor_t = tokenize(lens, corrupted)
    pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
    fig_patch = plot_patching_importance_bar(pr)

    summary = (
        "### Narrative arc\n"
        "1. **Structure** — which modules fire and tensor shapes.\n"
        "2. **Attention** — where the model looks on this prompt.\n"
        "3. **Logit lens** — how next-token predictions sharpen across depth.\n"
        "4. **Patching** — which sublayers causally restore the clean behavior.\n\n"
        + format_patching_summary_html(pr)
    )
    return fig_shape, fig_attn, fig_logit_hm, fig_logit_evo, fig_patch, summary
