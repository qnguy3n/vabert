"""
pipeline.py — VABERTPipeline: extracts visual acuity readings and commits
each field only when the text states it.

Each detected VA reading carries laterality, method and notation. For each
field the final value is one of:
  a value   the text states it: the model is confident (>= the field's
            threshold in thresholds.json) AND the rule-based evidence
            check (evidence_rules.py) reads the same value from the text
  'Unk'     the text does not state it (model and evidence check agree);
            a correct, final answer, not a request for review
  'Review'  the system isn't sure: low confidence, or model and evidence
            check disagree
The value itself also goes to review if it fails the notation's grammar
(e.g. "10/12", "6/52" as Snellen).

Letter status: 'complete' (every field committed), 'complete_with_unk'
(some fields Unk, none in review), 'needs_review', or 'no_va'.

Usage:
    from vabert import VABERTPipeline
    pipe = VABERTPipeline()
    result = pipe.run(text)   # {'status': ..., 'readings': [...]}
"""

import json, os

from .inference import VAModel, gate, FIELDS
from .postprocess import filter_implausible

# model/ and thresholds.json sit next to the package in the release folder.
# VABERT_MODEL_DIR and VABERT_THRESHOLDS point elsewhere, for example after a
# non-editable install that copies only the code.
RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VABERTPipeline:
    def __init__(self, model_dir=None, device=None, thresholds=None):
        model_dir = model_dir or os.environ.get('VABERT_MODEL_DIR') or os.path.join(RELEASE_DIR, 'model')
        weights = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"no checkpoint in {model_dir}; pass model_dir= or set VABERT_MODEL_DIR")
        with open(weights, 'rb') as f:
            if f.read(40).startswith(b'version https://git-lfs'):
                raise RuntimeError(f"{weights} is a Git LFS pointer, not the model weights: install Git LFS "
                                   "(https://git-lfs.com), then run `git lfs install` and `git lfs pull` in the repository")
        self.model = VAModel(model_dir, device=device)
        if thresholds is None:
            path = os.environ.get('VABERT_THRESHOLDS') or os.path.join(RELEASE_DIR, 'thresholds.json')
            thresholds = json.load(open(path))['tau']
        self.tau = thresholds

    def run(self, text):
        spans = filter_implausible(text, self.model.predict(text))
        spans = gate(text, spans, self.tau)
        readings = []
        for x in spans:
            row = {'va_value': x['val'], 'start': x['s'], 'end': x['e'],
                   'value_state': x['value_state'], 'value_reason': x['value_reason'],
                   'span_confidence': round(x['span_conf'], 4)}
            for f in FIELDS:
                ev = x.get(f + '_evidence')
                row[f] = x[f + '_final']
                row[f + '_state'] = x[f + '_state']
                row[f + '_reason'] = x[f + '_reason']
                row[f + '_model'] = x[f]
                row[f + '_confidence'] = round(x[f + '_conf'], 4)
                row[f + '_evidence'] = text[ev[0]:ev[1]] if ev else None
            row['label'] = f"VA-{row['lat']}-{row['method']}"
            readings.append(row)
        if not readings:
            status = 'no_va'
        elif any(r['value_state'] == 'review' or any(r[f + '_state'] == 'review' for f in FIELDS) for r in readings):
            status = 'needs_review'
        elif any(r[f + '_state'] == 'unk' for r in readings for f in FIELDS):
            status = 'complete_with_unk'
        else:
            status = 'complete'
        return {'status': status, 'readings': readings}
