"""Activation patching importance plots."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.schemas import patching_dict_to_viz

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def plot_patching_importance_bar(
    patching_result: Dict[str, Any],
    *,
    use_normalized: bool = True,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 560,
) -> "go.Figure":
    """
    Horizontal bar chart of patch effect per module (normalized or raw).
    """
    v = patching_dict_to_viz(patching_result)
    vals = v.normalized_effects if use_normalized else v.effects
    labels = [truncate_label(m, max_len=48) for m in v.module_names]

    # Sort by magnitude
    order = np.argsort(np.abs(vals))[::-1]
    labels = [labels[i] for i in order]
    vals = vals[order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=np.where(vals >= 0, "#16a34a", "#dc2626"),
            hovertemplate="module=%{y}<br>effect=%{x:.4f}<extra></extra>",
        )
    )
    ylab = "Normalized effect" if use_normalized else "Effect (metric delta)"
    fig.update_xaxes(title_text=ylab)
    t = title or "Activation patching — per-module causal effect"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_patching_importance_heatmap(
    patching_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 820,
    height: int = 400,
) -> "go.Figure":
    """Single-row heatmap of normalized effects (compact slide-friendly)."""
    v = patching_dict_to_viz(patching_result)
    labels = [truncate_label(m, max_len=32) for m in v.module_names]
    z = v.normalized_effects.reshape(1, -1)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=["effect"],
            colorscale="RdYlGn",
            zmid=0.0,
            hovertemplate="%{x}<br>norm_effect=%{z:.4f}<extra></extra>",
        )
    )
    t = title or "Patching — normalized effect heatmap"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_yaxes(showticklabels=False)
    return fig


def format_patching_summary_html(patching_result: Dict[str, Any]) -> str:
    """Compact HTML summary for Gradio Markdown."""
    c = float(patching_result["clean_metric"])
    r = float(patching_result["corrupted_metric"])
    te = float(patching_result["total_effect"])
    return (
        f"<div style='font-family:system-ui;line-height:1.5'>"
        f"<b>Clean metric:</b> {c:.4f}<br/>"
        f"<b>Corrupted metric:</b> {r:.4f}<br/>"
        f"<b>Total effect (corrupted − clean):</b> {te:.4f}"
        f"</div>"
    )
