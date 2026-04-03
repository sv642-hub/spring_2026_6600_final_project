# ModelLens visualization layer

## Design choices

1. **Analysis modules** return dicts with a stable **v1 contract** for keys used by plots (`token_labels`, `layers_ordered`, etc.). Legacy keys (`layer_results`, `attention_maps`) are preserved so existing scripts keep working.

2. **Visualization** stays separate from analysis: plot functions accept either raw analysis dicts or small **dataclass views** built via `modellens.visualization.schemas` helpers.

3. **Plotly** is the default renderer (notebooks + Gradio `Plotly` components). Figures use a shared template from `common.default_plotly_layout`.

4. **Shape trace** is computed after a forward with hooks on all modules; ordering follows `named_modules()` traversal (parent before children), matching `HookManager.attach_all` behavior.

5. **Activation patching** no longer calls `_forward_hooks.clear()` on every submodule (that broke unrelated hooks). It calls `ModelLens.clear()` when a `lens` is passed, removing only ModelLens-managed hooks.
