"""Normalize Hugging Face tokenizer outputs (e.g. BatchEncoding) for ``model(**kwargs)``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import torch


def hf_inputs_to_dict(inputs: Any) -> Dict[str, torch.Tensor]:
    """
    Convert tokenizer output or a raw tensor into a plain ``dict`` of tensors.

    ``BatchEncoding`` is not always ``isinstance(..., dict)`` across transformers
    versions; passing it through the wrong branch can set ``input_ids`` to the
    whole encoding object.
    """
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if hasattr(inputs, "input_ids"):
        d: Dict[str, torch.Tensor] = {"input_ids": inputs["input_ids"]}
        for k in ("attention_mask", "token_type_ids"):
            if k in inputs:
                d[k] = inputs[k]
        return d
    return {"input_ids": inputs}
