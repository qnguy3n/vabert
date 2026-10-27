# vabert 1.0.0

Extracts visual acuity (VA) readings from UK ophthalmology clinic letters. For each reading it reports the value, the eye, the measurement method and the notation, and it commits an eye or method only when the letter states it. Everything needed to run it (code, model, thresholds) is in this folder; nothing is downloaded at run time.

## What it returns

For each reading, three fields:

| Field | Values |
|---|---|
| `lat` (eye) | `Right`, `Left`, `Both`, `Unk` |
| `method` | `Aided`, `Unaided`, `AidedPH`, `UnaidedPH`, `UnkPH`, `Unk` (PH = pinhole) |
| `notation` | `snellen_m`, `snellen_ft`, `etdrs`, `logmar`, `decimal`, `thomson`, `qualitative`, `Unk` |

A field is committed only when the model is confident and a rule-based check of the text agrees. `Unk` means the letter doesn't state it, which is a final answer. `Review` means the pipeline isn't sure and a person should check; the model's own answer is still reported. Each letter gets a status: `complete`, `complete_with_unk`, `needs_review` or `no_va`.

## Install

From this folder, into the Python environment you want to use (Python 3.9+, PyTorch and transformers):

```
pip install -e .
```

The `-e` (editable) install keeps the package pointing at this folder, so it finds `model/` and `thresholds.json` here. Nothing is downloaded at run time. Without installing, both the Python import and `python -m vabert` work when run from this folder.

## Use from Python

```python
from vabert import VABERTPipeline

pipe = VABERTPipeline()                  # loads the model once; GPU if available
result = pipe.run("Visual acuity with glasses: right eye 6/9, left eye 6/12.")
result['status']                         # 'complete', 'complete_with_unk', 'needs_review' or 'no_va'
for r in result['readings']:
    print(r['va_value'], r['lat'], r['method'], r['notation'])   # 6/9 Right Aided snellen_m ...
```

Each reading also has `start`/`end` offsets, `label` (`VA-<lat>-<method>`), and for each of `lat`, `method`, `notation` its state (`committed`, `unk`, `review`), reason, the model's own answer and confidence, and the evidence text. `VABERTPipeline(model_dir=..., device='cpu', thresholds={...})` overrides the defaults. `usage.ipynb` shows parsed tables, raw predictions with confidence scores, and examples of each case.

## Use from the command line

```
vabert --text "Visual acuity: 85 on the right and 79 on the left."
vabert --file letter.txt --json                      # full result as JSON
cat letter.txt | vabert --stdin
vabert --corpus letters.csv --text-column letter_text --id-column id --out results.csv
vabert --help
```

`--corpus` writes one CSV row per reading (and one row for each letter without VA), with the letter status, final values, states, reasons, model answers, confidences and evidence. `--device cpu` and `--model-dir` work as in Python. Without installing, use `python -m vabert ...` from this folder.

## License and changes

Proprietary, all rights reserved; see `LICENSE`. Release history is in `CHANGELOG.md`.
