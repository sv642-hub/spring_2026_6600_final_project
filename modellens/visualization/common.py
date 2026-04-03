"""Shared helpers for tensor handling, styling, and safe plotting."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

import torch


def to_numpy(
    x: Union[torch.Tensor, np.ndarray, Any],
    *,
    reduce_batch: bool = True,
) -> np.ndarray:
    """Convert tensor or array to numpy float64. Handles batch dim [0]."""
    if x is None:
        raise ValueError("to_numpy received None")
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if reduce_batch and t.dim() >= 1 and t.shape[0] == 1:
            t = t[0]
        return t.float().numpy()
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float64)
        if reduce_batch and arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    return np.asarray(x, dtype=np.float64)


def tensor_to_dataframe(
    tensor: torch.Tensor,
    *,
    index: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
) -> "pd.DataFrame":
    """2D tensor -> DataFrame with optional index/column labels."""
    if pd is None:
        raise ImportError("pandas is required for tensor_to_dataframe; pip install pandas")
    arr = to_numpy(tensor, reduce_batch=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {arr.shape}")
    idx = index if index is not None else [str(i) for i in range(arr.shape[0])]
    cols = columns if columns is not None else [str(i) for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


def truncate_label(text: str, max_len: int = 24, suffix: str = "…") -> str:
    """Shorten long token or module names for axis labels."""
    if len(text) <= max_len:
        return text
    keep = max(1, max_len - len(suffix))
    return text[:keep] + suffix


def truncate_labels(labels: Sequence[str], max_len: int = 24) -> list[str]:
    return [truncate_label(str(t), max_len=max_len) for t in labels]


def default_plotly_template() -> str:
    return "plotly_white"


def default_plotly_layout(
    *,
    title: str,
    width: Optional[int] = 900,
    height: Optional[int] = 500,
    font_size: int = 12,
) -> dict:
    """Base layout dict merged into Plotly figures."""
    return {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "template": default_plotly_template(),
        "font": {"size": font_size},
        "margin": {"l": 60, "r": 40, "t": 60, "b": 80},
        "width": width,
        "height": height,
    }


def safe_int_list(seq: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for x in seq:
        if hasattr(x, "item"):
            out.append(int(x.item()))
        else:
            out.append(int(x))
    return out


def format_prob(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"
