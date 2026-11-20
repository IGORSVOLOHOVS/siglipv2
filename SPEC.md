# siglipv2 - software requirements specification

Reverse-engineered on 2026-11-18 from the code at commit `bd514f2` (2026-11-08). It describes what the code does, not what it was meant to do. Each requirement names the line it was read from and the test that checks it, if any. The documentation hub is [`docs/README.md`](docs/README.md).

<!-- jira-sync:start -->
Jira: KAN-90

## Tasks

This list is one ticket in Jira project KAN, [KAN-90](https://igorsvolohovs.atlassian.net/browse/KAN-90). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] Model loading (section 3.1) <!-- spec-model-loading -->
- [x] Classification (section 3.2) <!-- spec-classification -->
- [x] Console entry point (section 3.3) <!-- spec-console-entry-point -->
- [x] Logging (section 3.4) <!-- spec-logging -->
- [x] Packaging (section 3.5) <!-- spec-packaging -->
- [ ] Alternative classification backends behind the `Classifier` protocol. (declared, not implemented) <!-- spec-alternative-classification-backends-behind-the-cla -->

<!-- jira-sync:end -->

## 1. Purpose and scope

The system classifies one image zero-shot against a list of text labels with the SigLIP 2 model `google/siglip2-base-patch16-224` and prints the best-scoring label (`classify_siglip.py:25-74`, `README.md:3`).

In scope:

- loading the model and its processor through Hugging Face `transformers`, on CUDA or CPU;
- turning labels into prompts, scoring one image against them, returning the top-1 index;
- the console entry point `main()` with fixed inputs, and its output and error handling;
- packaging the module as a wheel.

Out of scope (not present in the code): training or fine-tuning (`README.md:3`), command-line options, batches of images, printing scores, any network service, any file output.

## 2. Context

| Item | Description | Source |
| --- | --- | --- |
| Actor: person at a terminal | Starts `.\recipe.ps1 run` from the repository root and reads the printed label. Who these people are is not stated (see section 9). | `README.md:10`, `docs/manual.md:7` |
| Actor: Python caller | May construct `SigLIPClassifier` and call `classify()`; no such caller exists in the repository apart from the tests. | `classify_siglip.py:20-59` |
| External system: Hugging Face Hub | Source of the model weights and processor files, downloaded by `transformers` on the first run into the Hugging Face cache. | `classify_siglip.py:28-29`, `docs/install.md:9` |
| Runtime | CPython 3.13 in a uv-managed environment; PyTorch on CUDA when available, otherwise CPU; on Windows the locked `torch` wheel is the CPU build. | `pyproject.toml:21`, `classify_siglip.py:26`, `docs/install.md:7` |
| Build and CI | `recipe.ps1` verbs; Gitea Actions workflows on a runner labelled `windows`. | `recipe.ps1:47-110`, `.gitea/workflows/propose.yaml:14` |

## 3. Functional requirements

### 3.1 Model loading

- **FR-1.** The system shall load the model with `AutoModel.from_pretrained(model_name)` and the processor with `AutoProcessor.from_pretrained(model_name)`, where `model_name` defaults to `google/siglip2-base-patch16-224`.
  Source: `classify_siglip.py:25`, `classify_siglip.py:28-29`.
  Verification: `test_the_constructor_loads_the_named_model_and_its_processor_on_the_cpu`, `test_the_constructor_picks_cuda_when_it_is_there`.
- **FR-2.** The system shall select the device `cuda` when `torch.cuda.is_available()` returns true and `cpu` otherwise, and move the model to that device.
  Source: `classify_siglip.py:26`, `classify_siglip.py:28`.
  Verification: `test_the_constructor_loads_the_named_model_and_its_processor_on_the_cpu`, `test_the_constructor_picks_cuda_when_it_is_there`.

### 3.2 Classification

- **FR-3.** The system shall turn every label, in the given order, into the prompt `This is a photo of <label>.`
  Source: `classify_siglip.py:37`.
  Verification: `test_every_label_becomes_a_prompt_and_the_image_arrives_as_rgb`.
- **FR-4.** The system shall open the image at `image_path` with Pillow and convert it to RGB.
  Source: `classify_siglip.py:43`.
  Verification: `test_every_label_becomes_a_prompt_and_the_image_arrives_as_rgb`.
- **FR-5.** When the image file does not exist, the system shall log at ERROR `Image not found: <path>` and re-raise `FileNotFoundError` without calling the processor.
  Source: `classify_siglip.py:42-46`.
  Verification: `test_a_missing_image_is_logged_and_raised`.
- **FR-6.** The system shall call the processor with `text=<prompts>`, `images=<image>`, `padding="max_length"`, `return_tensors="pt"` and move the resulting inputs to the selected device.
  Source: `classify_siglip.py:48-50`.
  Verification: `test_every_label_becomes_a_prompt_and_the_image_arrives_as_rgb`.
- **FR-7.** The system shall run the model with gradient computation disabled.
  Source: `classify_siglip.py:52-53`.
  Verification: `test_the_index_of_the_highest_score_is_returned`.
- **FR-8.** The system shall return, as a Python `int`, the index of the maximum of `logits_per_image` along its last dimension.
  Source: `classify_siglip.py:56-59`.
  Verification: `test_the_index_of_the_highest_score_is_returned`.

### 3.3 Console entry point

- **FR-9.** `main()` shall classify `data/image.png` against the labels `a car`, `a bicycle`, `a truck` with the model `google/siglip2-base-patch16-224`.
  Source: `classify_siglip.py:64-71`.
  Verification: `test_main_prints_the_index_and_the_label`.
- **FR-10.** `main()` shall print `Predicted index: <index>` and `Predicted label: <label>` to standard output through a `rich` console, with bold green markup.
  Source: `classify_siglip.py:9`, `classify_siglip.py:73-74`.
  Verification: `test_main_prints_the_index_and_the_label`.
- **FR-11.** When any `Exception` is raised while loading or classifying, `main()` shall log at ERROR `An error occurred: <message>`, print no prediction and return normally, so the process exits 0.
  Source: `classify_siglip.py:69-77`, `docs/manual.md:18`.
  Verification: `test_main_reports_a_failure_instead_of_raising` (no test checks the exit code).
- **FR-12.** The module shall call `main()` when it is executed as a script.
  Source: `classify_siglip.py:80-81`.
  Verification: no test.

### 3.4 Logging

- **FR-13.** On import the module shall configure logging at level INFO with the format `%(levelname)s: %(message)s`.
  Source: `classify_siglip.py:12`.
  Verification: no test.
- **FR-14.** The system shall log at INFO `Loading model <name> on <device>...` before loading and `Classifying <path>` before opening the image, and log the prompt list at DEBUG.
  Source: `classify_siglip.py:27`, `classify_siglip.py:39-40`.
  Verification: no test.

### 3.5 Packaging

- **FR-15.** The build shall produce a wheel that contains the single module `classify_siglip` and not `data/image.png`.
  Source: `pyproject.toml:42-43`, `recipe.ps1:19`, `recipe.ps1:48-50`, `docs/install.md:11`.
  Verification: no test.

### 3.6 Declared but not implemented

- Alternative classification backends behind the `Classifier` protocol. The pre-lab README (commit `22232b9`, section "Architecture") described the protocol as ensuring "future-proofing for alternative classification backends". The protocol exists (`classify_siglip.py:16-17`), but there is no second implementation and no code accepts a `Classifier`.

No `TODO` or `FIXME` markers exist in `classify_siglip.py`, `tests/`, `pyproject.toml`, `recipe.ps1` or the docs (checked with `grep`).

## 4. Interfaces

### 4.1 Command line

| Invocation | Arguments | Standard output | Standard error | Exit code | Source |
| --- | --- | --- | --- | --- | --- |
| `.\recipe.ps1 run` -> `uv run --frozen python classify_siglip.py` | none read; extra arguments are forwarded by the recipe and ignored by the script | two `Predicted ...` lines, or nothing on failure | log lines (Python `logging.basicConfig` writes to stderr by default) | 0 on success and on a handled failure | `recipe.ps1:60-62`, `classify_siglip.py:62-81`, `docs/manual.md:18` |

### 4.2 Python API

| Element | Signature | Behaviour | Source |
| --- | --- | --- | --- |
| `Classifier` | `Protocol`: `classify(self, image_path: str, labels: list[str]) -> int` | Structural type; unused by the code | `classify_siglip.py:16-17` |
| `SigLIPClassifier.__init__` | `(model_name: str = "google/siglip2-base-patch16-224")` | FR-1, FR-2; sets `device`, `model`, `processor` | `classify_siglip.py:25-29` |
| `SigLIPClassifier.classify` | `(image_path: str, labels: list[str]) -> int` | FR-3 to FR-8 | `classify_siglip.py:31-59` |
| `main` | `() -> None` | FR-9 to FR-11 | `classify_siglip.py:62-77` |

### 4.3 Messages

| Level / stream | Text | Source |
| --- | --- | --- |
| INFO | `Loading model <name> on <device>...` | `classify_siglip.py:27` |
| INFO | `Classifying <path>` | `classify_siglip.py:39` |
| DEBUG (not shown at the configured level) | `Using prompts: [...]` | `classify_siglip.py:40` |
| ERROR | `Image not found: <path>` | `classify_siglip.py:45` |
| ERROR | `An error occurred: <message>` | `classify_siglip.py:77` |
| stdout | `Predicted index: <index>` / `Predicted label: <label>` | `classify_siglip.py:73-74` |

### 4.4 Build verbs

| Verb | Command | Source |
| --- | --- | --- |
| `build` | `uv build --wheel --out-dir dist` | `recipe.ps1:48-50` |
| `run` | `uv run --frozen python classify_siglip.py` | `recipe.ps1:60-62` |
| `test` | version check, `pytest`, `mypy`, `ruff check .`, `ruff format .` | `recipe.ps1:82-108` |

The recipe is the repo-quality-template one; apart from the metadata lines (`recipe.ps1:11-24`) only the `run` verb differs from the template's (compared with `recipe.ps1` on the template's `release` branch). The other verbs (`clean`, `info`, `install`, `uninstall`, `publish`) are the template's, unchanged, and are not specified here.

## 5. Data

| Entity | Format and location | Source |
| --- | --- | --- |
| Input image | `data/image.png`, PNG 1920x1440, 8-bit RGB (read with `file`), stored in Git LFS | `.gitattributes:9`, `docs/install.md:4` |
| Labels | Python `list[str]` in `main()` | `classify_siglip.py:65` |
| Model weights and processor files | Hugging Face cache, filled on the first run | `docs/install.md:9` |
| Result | `int` index into the labels; only printed, not stored | `classify_siglip.py:57-59`, `:73-74` |
| Manual-test list | `tests/manual/tests.json`: `{"version": 1, "tests": []}`; entries need `id` (matching `[a-z0-9][a-z0-9_-]{0,60}`, not `verdict`), `title`, `expected`, `steps`, optional `issue`, `blocking`, `needs`; UTF-8 without BOM | `tests/manual/tests.json:1-4`, `tests/test_manual_list.py:9-51` |
| Build artefact | wheel in `dist\` | `recipe.ps1:49` |

The script writes no files (inference: no file-writing call in `classify_siglip.py`).

## 6. Non-functional requirements

- **NFR-1.** The project shall run on CPython 3.13 only. Source: `pyproject.toml:21`, `uv.lock:3`.
- **NFR-2.** Every direct runtime and development dependency, and the build backend, shall be pinned to an exact version, and the recipe's `run` and `test` verbs shall use the lock unchanged (`--frozen`). Source: `pyproject.toml:10`, `pyproject.toml:22-40`, `recipe.ps1:61`, `recipe.ps1:99-106`.
- **NFR-3.** The test suite shall not download model weights: `HF_HUB_OFFLINE=1` is set before `huggingface_hub` is imported, and the model and processor are replaced by fakes. Source: `tests/conftest.py:6`, `tests/test_classify_siglip.py:14-59`. Verification: `test_the_hub_is_offline_for_the_whole_run`.
- **NFR-4.** Branch coverage of the project code shall be at least 95%. Source: `pyproject.toml:75-76`, `pyproject.toml:78-82`.
- **NFR-5.** `classify_siglip.py` shall pass `mypy --strict`, with untyped calls into `transformers` as the one exclusion. Source: `pyproject.toml:87-94`.
- **NFR-6.** The code shall pass `ruff check` with the rule sets `E, W, F, I, UP, B, BLE, SIM, C90, S, RUF` (except `S101`) and `ruff format` with line length 100 and LF endings. Source: `pyproject.toml:49-68`.
- **NFR-7.** GPU use is optional: the code falls back to CPU (FR-2); on Windows the one lock file resolves to the CPU `torch` wheel, on Linux it adds CUDA 13 packages. Source: `pyproject.toml:23-24`, `uv.lock:896-911`, `docs/install.md:7`.
- **NFR-8.** `classify_siglip.py` shall need no credentials: it reads no environment variables (checked with `grep` for `environ` and `getenv`). No key is committed to the repository. Source: `classify_siglip.py`, `CLAUDE.md` rule 4, `.gitignore:20-24`.
- **NFR-9.** The repository follows the lab tooling of repo-quality-template; `README.md:21` states compliance with revision 4.0.0 of the lab requirements except R-02, R-09, R-37, R-41, R-42 (issues #7-#11). The tooling itself is specified by that template, not here.

## 7. Constraints and dependencies

| Package | Version | Kind | Source |
| --- | --- | --- | --- |
| torch | 2.11.0 | runtime | `pyproject.toml:25` |
| transformers | 5.5.4 | runtime | `pyproject.toml:26` |
| pillow | 12.2.0 | runtime | `pyproject.toml:27` |
| rich | 15.0.0 | runtime | `pyproject.toml:28` |
| ruff | 0.16.1 | dev | `pyproject.toml:35`, `pyproject.toml:51` |
| mypy | 2.3.0 | dev | `pyproject.toml:36` |
| pytest | 8.4.2 | dev | `pyproject.toml:37` |
| pytest-cov | 5.0.0 | dev | `pyproject.toml:38` |
| coverage | 7.15.2 | dev | `pyproject.toml:39` |
| setuptools | 80.9.0 | build backend | `pyproject.toml:10` |
| huggingface-hub | 1.33.0 | transitive (via transformers), imported directly by a test | `uv.lock:257-258`, `tests/test_no_weights_are_downloaded.py:3` |

Other constraints: uv is required but its version is not pinned (`recipe.ps1:16`); Git LFS is needed to get the image (`docs/install.md:4`); `recipe.ps1` is a PowerShell script; licence MIT (`LICENSE`, `pyproject.toml:19`).

## 8. Acceptance

1. `uv sync --frozen` completes on Python 3.13 (`docs/install.md:5`).
2. `.\recipe.ps1 test` exits 0. This checks FR-1 to FR-11 and NFR-3 to NFR-6 (`recipe.ps1:82-108`).
3. With network access, `.\recipe.ps1 run` from the repository root prints `Predicted index: <0, 1 or 2>` and the matching `Predicted label: ...` line (FR-9, FR-10). The pre-lab README at commit `22232b9` recorded `Predicted index: 0` / `Predicted label: a car` for `data/image.png`; this was not re-run for this document.
4. With `data/image.png` absent, the same run logs `Image not found: data/image.png` and `An error occurred: ...`, prints no prediction, and exits 0 (FR-5, FR-11).
5. `.\recipe.ps1 build` leaves a wheel in `dist\` whose only module is `classify_siglip.py` (FR-15).
6. FR-12 to FR-14 have no automated test; check them against the log of step 3.

## 9. Open questions

1. Who the users are and what the project is for beyond the demonstration; commit `7efb2d1` indicates it started from a course lesson.
2. What `data/image.png` depicts and which label is the expected answer; no test checks the real model's result.
3. No accuracy, latency or memory targets exist anywhere in the repository.
4. Is exit code 0 on failure (FR-11) intended? `docs/manual.md:18` documents it without a reason.
5. Should the model be pinned to a Hugging Face `revision`? `from_pretrained` is called without one (`classify_siglip.py:28-29`).
6. Should the script accept command-line options? `recipe.ps1 run` already forwards arguments (`recipe.ps1:61`) that nothing reads.
7. Is CUDA on Windows wanted? The code supports it (FR-2), but on Windows the lock installs the CPU build (`docs/install.md:7`). Partly answered: `pyproject.toml:23-24` says the CPU build is accepted so that the Windows runner never fetches a CUDA stack; whether a Windows user should get CUDA is not stated.
8. Does a Gitea runner labelled `windows` serve this repository? `CLAUDE.md` provides for the case where none does.
9. What exceptions R-37, R-41 and R-42 (issues #9-#11) consist of; the repository files describe only R-02 and R-09 (`CLAUDE.md`), and the requirements document is linked at a loopback address (`README.md:21`).
10. Whether `.agent/` (vendored agent kit, C++ and Python) is still wanted; nothing in `recipe.ps1` or the tests uses it. Partly answered: it was kept knowingly when the lab tooling came in - `pyproject.toml:54-57` calls it "vendored agent tooling" and excludes it from ruff, and coverage omits it (`pyproject.toml:82`).
11. Whether `huggingface_hub`, imported by a test, should be a declared dependency.

## Screenshots of requirements

### FR-1 - The system shall load the model with AutoModel.frompretrained(modelname) and the processor with AutoProcessor.…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-2 - The system shall select the device cuda when torch.cuda.isavailable() returns true and cpu otherwise, and move…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-3 - The system shall turn every label, in the given order, into the prompt This is a photo of <label>

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-4 - The system shall open the image at imagepath with Pillow and convert it to RGB

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-5 - When the image file does not exist, the system shall log at ERROR Image not found: <path> and re-raise FileNot…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-6 - The system shall call the processor with text=<prompts>, images=<image>, padding="maxlength", returntensors="p…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-7 - The system shall run the model with gradient computation disabled

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-8 - The system shall return, as a Python int, the index of the maximum of logitsperimage along its last dimension

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-9 - main() shall classify data/image.png against the labels a car, a bicycle, a truck with the model google/siglip…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-10 - main() shall print Predicted index: <index> and Predicted label: <label> to standard output through a rich con…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-11 - When any Exception is raised while loading or classifying, main() shall log at ERROR An error occurred: <messa…

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-12 - The module shall call main() when it is executed as a script

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-13 - On import the module shall configure logging at level INFO with the format %(levelname)s: %(message)s

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-14 - The system shall log at INFO Loading model <name> on <device>... before loading and Classifying <path> before …

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.

### FR-15 - The build shall produce a wheel that contains the single module classifysiglip and not data/image.png

> Screenshot skipped: the code does not run here - the project runs on PyTorch-class libraries and trained weights, and torch is not installed here (multi-gigabyte); requirements whose code does not need torch will be revisited in a later pass.
