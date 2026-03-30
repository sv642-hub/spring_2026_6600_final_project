#!/usr/bin/env python3
"""
Generate core Plotly figures (HTML) for a quick presentation dry run.

Requires: pip install -e ".[viz]"
First run downloads a small HF model (default: gpt2).

Usage:
  python examples/quick_viz_demo.py
  python examples/quick_viz_demo.py --out ./viz_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from modellens import ModelLens
from modellens.analysis.attention import run_attention_analysis
from modellens.analysis.logit_lens import run_logit_lens
from modellens.visualization.attention import plot_attention_heatmap
from modellens.visualization.logit_lens import (
    plot_logit_lens_evolution,
    plot_logit_lens_heatmap,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", type=Path, default=Path("viz_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager"
    )
    model.eval()
    lens = ModelLens(model)
    lens.adapter.set_tokenizer(tok)
    prompt = "The capital of France is"
    tokens = tok(prompt, return_tensors="pt")

    attn = run_attention_analysis(lens, tokens)
    plot_attention_heatmap(attn, layer_index=0, head_index=0).write_html(
        args.out / "attention.html"
    )

    ll = run_logit_lens(lens, tokens, tokenizer=tok, top_k=5)
    plot_logit_lens_evolution(ll).write_html(args.out / "logit_evolution.html")
    plot_logit_lens_heatmap(ll).write_html(args.out / "logit_heatmap.html")

    print(f"Wrote HTML files to {args.out.resolve()}")


if __name__ == "__main__":
    main()
