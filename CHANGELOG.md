# Changelog

All notable changes to vabert are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-10-27

### Added

- README section on getting the code and model: installing Git LFS, cloning, and `git lfs pull` when the weights arrive as a pointer file.

### Fixed

- A clone made without Git LFS leaves a pointer file in place of `model/pytorch_model.bin`; loading it gave a misleading PyTorch unpickling error. vabert now stops with a message saying to install Git LFS and run `git lfs pull`.

## [1.0.0] - 2026-10-27

First official release.

### Added

- `VABERTPipeline` (`from vabert import VABERTPipeline`): extracts visual acuity readings from clinic letter text and reports, for each reading, the value, eye (`lat`), measurement method and notation, with character offsets.
- Explicit labels: eye `Right` / `Left` / `Both` / `Unk`; method `Aided` / `Unaided` / `AidedPH` / `UnaidedPH` / `UnkPH` / `Unk`; notation `snellen_m` / `snellen_ft` / `etdrs` / `logmar` / `decimal` / `thomson` / `qualitative` / `Unk`. `Unk` means the letter doesn't state it; no eye or method is assigned by convention.
- Commit gate: a field is committed only when the model's confidence reaches the field's threshold (`thresholds.json`) and a rule-based check of the text agrees; otherwise it is `Unk` (both agree the text is silent) or `Review`. The value itself goes to review if it doesn't fit its notation.
- Letter status: `complete`, `complete_with_unk`, `needs_review` or `no_va`.
- Filters that remove spans which aren't VA readings: blood pressure and follow-up intervals written like Snellen fractions, and word or fraction fragments.
- Command-line interface: `vabert` (or `python -m vabert`) with `--text`, `--file`, `--stdin` and `--corpus` (CSV in, one row per reading out), plus `--json`, `--out`, `--model-dir` and `--device`.
- Runs offline: the model is built from the bundled checkpoint and `bert_config.json`, with no download at run time. `VABERT_MODEL_DIR` and `VABERT_THRESHOLDS` point to a checkpoint and thresholds stored elsewhere.
- `usage.ipynb` with worked examples.
- `LICENSE` (proprietary, all rights reserved) and this changelog.
