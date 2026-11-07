# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13. The pins are locked for 3.13 only.
2. Install [Git LFS](https://git-lfs.com/): `data/image.png` is stored in it.
3. Clone this repository and run `uv sync --frozen` at its root.

On Windows the locked `torch` is the CPU build from PyPI, so the script runs on the CPU there.

The first `.\recipe.ps1 run` downloads `google/siglip2-base-patch16-224`, a few hundred megabytes, into the Hugging Face cache, so it needs a network connection. `.\recipe.ps1 test` needs none of that: `tests/conftest.py` sets `HF_HUB_OFFLINE`, the tests replace the model and the processor, and a test that reached for the weights would fail instead of fetching them.

`.\recipe.ps1 build` packs `classify_siglip.py` into a wheel in `dist\`. `data/image.png` is not in the wheel: start the script from a clone, with `.\recipe.ps1 run`.
