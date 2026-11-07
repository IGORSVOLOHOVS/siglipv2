# siglipv2

A console script that classifies an image zero-shot: it scores the image against a list of text labels with Google's SigLIP 2 (`google/siglip2-base-patch16-224`) and prints the label that fits best. Nothing is trained.

`classify_siglip.py` holds all of it. `SigLIPClassifier` loads the model and its processor, on CUDA when there is one and on the CPU otherwise, turns each label into the prompt `This is a photo of <label>.`, and returns the index of the highest score. `main()` classifies `data/image.png` against `a car`, `a bicycle` and `a truck`.

## Use

```powershell
.\recipe.ps1 run
```

The image, the labels and the model are fixed in `main()`; the script takes no options. `docs/install.md` says what to install and `docs/manual.md` what a run prints. `.\recipe.ps1 test` runs the tests, mypy and ruff, with the model replaced: no test downloads weights.

## Licence

MIT, see `LICENSE`.

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#7), R-09 (#8), R-37 (#9), R-41 (#10), R-42 (#11).
