"""Logit lens evolution charts from ``run_logit_lens`` outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def _layers_and_top(
    logit_lens_result: Dict[str, Any],
) -> tuple[List[str], List[List[str]], List[List[float]]]:
    layers = logit_lens_result.get("layers_ordered")
    if not layers:
        layers = list(logit_lens_result.get("layer_results", {}).keys())

    if "top_tokens_per_layer" in logit_lens_result:
        return (
            layers,
            logit_lens_result["top_tokens_per_layer"],
            logit_lens_result["top_probs_per_layer"],
        )

    lr = logit_lens_result["layer_results"]
    toks: List[List[str]] = []
    probs: List[List[float]] = []
    for name in layers:
        idx = lr[name]["top_k_indices"][0]
        pr = lr[name]["top_k_probs"][0]
        toks.append([str(int(idx[i].item())) for i in range(idx.shape[0])])
        probs.append([float(pr[i].item()) for i in range(pr.shape[0])])
    return layers, toks, probs


def plot_logit_lens_evolution(
    logit_lens_result: Dict[str, Any],
    *,
    rank_index: int = 0,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """
    Line chart: probability of the *rank_index*-th predicted token at each layer.

    rank_index 0 follows the top-1 candidate's probability through layers.
    """
    layers, _, probs = _layers_and_top(logit_lens_result)
    if not layers:
        raise ValueError("No layer data for logit lens plot")

    x = [truncate_label(L.replace(".", " / "), max_len=40) for L in layers]
    y = [p[rank_index] if rank_index < len(p) else float("nan") for p in probs]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=3, color="#2563eb"),
            marker=dict(size=8),
            hovertemplate="layer=%{x}<br>p=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Probability", range=[0, 1.05])
    fig.update_xaxes(title_text="Layer")
    t = title or f"Logit lens — top-{rank_index + 1} token probability by layer"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_logit_lens_heatmap(
    logit_lens_result: Dict[str, Any],
    *,
    top_ranks: int = 5,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 560,
) -> "go.Figure":
    """
    Heatmap: layers × rank slot (1..K) showing probability mass for top-k predictions.
    """
    layers, _, probs = _layers_and_top(logit_lens_result)
    k = min(top_ranks, len(probs[0]) if probs else 0)
    if k == 0:
        raise ValueError("No probability rows")

    z = np.array([p[:k] for p in probs], dtype=np.float64)
    y_labels = [truncate_label(L, max_len=36) for L in layers]
    x_labels = [f"rank {i + 1}" for i in range(k)]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            hovertemplate="layer=%{y}<br>%{x}<br>p=%{z:.4f}<extra></extra>",
        )
    )
    t = title or "Logit lens — top-k probabilities across layers"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_logit_lens_top_token_bars(
    logit_lens_result: Dict[str, Any],
    *,
    layer_index: int = -1,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 480,
) -> "go.Figure":
    """Horizontal bar chart of top-k tokens at one layer (default: last)."""
    layers, toks, probs = _layers_and_top(logit_lens_result)
    if not layers:
        raise ValueError("No layers")
    li = layer_index if layer_index >= 0 else len(layers) - 1
    labels = toks[li]
    p = probs[li]

    fig = go.Figure(
        go.Bar(
            x=p,
            y=labels,
            orientation="h",
            marker_color="#0d9488",
            hovertemplate="token=%{y}<br>p=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Probability", range=[0, 1.05])
    t = title or f"Top-k at layer — {truncate_label(layers[li], 48)}"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig
