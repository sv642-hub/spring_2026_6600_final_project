"""Lightweight views over analysis dicts for typing and Gradio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AttentionVizData:
    """Normalized attention payload for plotting."""

    weights: np.ndarray  # (heads, seq, seq) or (seq, seq)
    token_labels: List[str]
    layer_key: str
    num_heads: int
    seq_length: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogitLensVizData:
    """Per-layer top-k strings and probabilities for evolution plots."""

    layers_ordered: List[str]
    layer_labels_short: List[str]
    top_tokens: List[List[str]]  # len = n_layers, each len = k
    top_probs: List[List[float]]
    focus_token: Optional[str] = None


@dataclass
class PatchingVizData:
    """Activation patching summary for bar/heatmap plots."""

    module_names: List[str]
    effects: np.ndarray  # raw effect per module
    normalized_effects: np.ndarray
    clean_metric: float
    corrupted_metric: float
    total_effect: float


@dataclass
class ResidualVizData:
    layers: List[str]
    delta_norm: List[float]
    cosine_similarity: List[float]
    relative_contribution: List[float]


@dataclass
class EmbeddingVizData:
    token_labels: List[str]
    similarity_matrix: np.ndarray  # (seq, seq)
    norms: Optional[np.ndarray] = None


@dataclass
class ShapeTraceRow:
    module_name: str
    shape: tuple
    dtype: str


def patching_dict_to_viz(d: Dict[str, Any]) -> PatchingVizData:
    pe = d.get("patch_effects", {})
    names = list(pe.keys())
    if not names:
        return PatchingVizData(
            module_names=[],
            effects=np.array([], dtype=np.float64),
            normalized_effects=np.array([], dtype=np.float64),
            clean_metric=float(d.get("clean_metric", 0.0)),
            corrupted_metric=float(d.get("corrupted_metric", 0.0)),
            total_effect=float(d.get("total_effect", 0.0)),
        )
    effects = np.array([pe[k]["effect"] for k in names], dtype=np.float64)
    ne = np.array([pe[k]["normalized_effect"] for k in names], dtype=np.float64)
    return PatchingVizData(
        module_names=names,
        effects=effects,
        normalized_effects=ne,
        clean_metric=float(d["clean_metric"]),
        corrupted_metric=float(d["corrupted_metric"]),
        total_effect=float(d["total_effect"]),
    )


def residual_dict_to_viz(d: Dict[str, Any]) -> ResidualVizData:
    c = d.get("contributions", {})
    layers = list(c.keys())
    return ResidualVizData(
        layers=layers,
        delta_norm=[float(c[k]["delta_norm"]) for k in layers],
        cosine_similarity=[float(c[k]["cosine_similarity"]) for k in layers],
        relative_contribution=[float(c[k]["relative_contribution"]) for k in layers],
    )
