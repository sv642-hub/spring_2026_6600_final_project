"""
Presentation-ready Plotly figures built from ModelLens analysis outputs.

Design notes live in ``modellens/visualization/DESIGN.md``. Prefer importing
specific plot functions from submodules for tree-shaking in notebooks.
"""

from modellens.visualization.activation_patching import (
    format_patching_summary_html,
    plot_patching_importance_bar,
    plot_patching_importance_heatmap,
)
from modellens.visualization.attention import (
    plot_attention_head_grid,
    plot_attention_heatmap,
)
from modellens.visualization.common import (
    default_plotly_layout,
    default_plotly_template,
    tensor_to_dataframe,
    to_numpy,
    truncate_label,
    truncate_labels,
)
from modellens.visualization.embeddings import (
    plot_embedding_norms,
    plot_embedding_similarity_heatmap,
)
from modellens.visualization.logit_lens import (
    plot_logit_lens_evolution,
    plot_logit_lens_heatmap,
    plot_logit_lens_top_token_bars,
)
from modellens.visualization.residuals import (
    plot_residual_contributions,
    plot_residual_lines,
)
from modellens.visualization.schemas import (
    AttentionVizData,
    EmbeddingVizData,
    LogitLensVizData,
    PatchingVizData,
    ResidualVizData,
    patching_dict_to_viz,
    residual_dict_to_viz,
)
from modellens.visualization.shapes import (
    compute_shape_trace,
    plot_shape_trace_table,
    shape_trace_mermaid,
    shape_trace_to_dataframe,
)

__all__ = [
    "AttentionVizData",
    "EmbeddingVizData",
    "LogitLensVizData",
    "PatchingVizData",
    "ResidualVizData",
    "compute_shape_trace",
    "default_plotly_layout",
    "default_plotly_template",
    "format_patching_summary_html",
    "patching_dict_to_viz",
    "plot_attention_head_grid",
    "plot_attention_heatmap",
    "plot_embedding_norms",
    "plot_embedding_similarity_heatmap",
    "plot_logit_lens_evolution",
    "plot_logit_lens_heatmap",
    "plot_logit_lens_top_token_bars",
    "plot_patching_importance_bar",
    "plot_patching_importance_heatmap",
    "plot_residual_contributions",
    "plot_residual_lines",
    "plot_shape_trace_table",
    "residual_dict_to_viz",
    "shape_trace_mermaid",
    "shape_trace_to_dataframe",
    "tensor_to_dataframe",
    "to_numpy",
    "truncate_label",
    "truncate_labels",
]
