# Hyperflare Recipes

Official model recipes, reusable PyTorch and b12x model components, and
qualification drivers for preparing runtime cores served by
[Flare](https://github.com/NVIDIA/hyperflare-inference).

## Layout

- `hyperflare_recipes/recipes/` — official model recipes (`deepseek4`,
  `deepseek41`, `gemma4`, `glm53`, `inkling2` including `inkling2_mini`,
  `kimi3`, `kimi31`, `mimo_v2`, `minimax_m3`, `muse_spark`, `nemotron_h`,
  `qwen38`, `qwen38_dense`, `step5`).
- `hyperflare_recipes/components/` — reusable b12x and Triton model building
  blocks (weights, linear projections, MoE, attention, MLA, GDN, Mamba,
  speculative drafting, and head exchange).
- `hyperflare_recipes/fixtures/` — operator-tuning benchmark fixtures.
- `hyperflare_recipes/reference/` — numerical reference implementations.
- `hyperflare_recipes/cli.py` — `ignition` CLI entrypoint wrapping
  `ignition.cli.main(...)` from the core Flare repository.
- `qualify/`, `tests/`, `scripts/`, `tools/` — model qualification drivers and
  recipe test suites.

## Usage

In an environment with `ignition` (and `flare-heliopause`) installed from the
core Flare repository:

```sh
ignition export MODEL --gpus 0
```

All core preparation and publishing subcommands (`export`, `debug-serve`,
`bench`, `join`, `keygen`, `key-password`, `sign`, `edit`, `combine`,
`convert`) are provided by `ignition.cli` and inherited automatically.

## Building an Independent Recipe Repository

External recipe repositories do not need to merge into Flare or
`hyperflare-recipes`. Depend on `ignition` from Flare (and optionally
`hyperflare-recipes` to reuse b12x model components) and expose a CLI entrypoint
calling `ignition.cli.main`:

```python
from pathlib import Path
from ignition.cli import main as ignition_main
from .recipes import RECIPES

def main(argv=None) -> int:
    return ignition_main(
        argv,
        recipes=RECIPES,
        prog="my-recipes",
        source_roots=(("my_recipes", Path(__file__).resolve().parent),),
    )
```
