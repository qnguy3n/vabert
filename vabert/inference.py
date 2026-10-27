"""
inference.py — model inference (VAModel, BertForVAFactorized) and the
commit gate of the pipeline.

predict(text) returns spans with, per field (lat, method, notation), the
model's distribution, its argmax and confidence. gate() turns that into
the final per-field state:

  committed  value stated in the text, model confident (p >= tau) AND the
             rule-based evidence labeller (evidence_rules) agrees
  Unk        model and evidence labeller both say the text is silent
  Review     anything else: low confidence, or model and evidence disagree
"""

import json, os, re
import torch

from .evidence_rules import label_document
from . import model_def as M

MAX_LEN, STRIDE, BATCH = 512, 128, 16
FIELDS = ('lat', 'method', 'notation')


class VAModel:
    def __init__(self, model_dir, device=None):
        from transformers import AutoTokenizer, BertConfig
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # bert_config.json lets the backbone be built offline; without it the
        # pretrained BioBERT weights are fetched (or read from the HF cache) first
        cfg = os.path.join(model_dir, 'bert_config.json')
        self.model = M.BertForVAFactorized(config=BertConfig.from_json_file(cfg) if os.path.exists(cfg) else None)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

    def _char_probs(self, text):
        enc = self.tokenizer(text, max_length=MAX_LEN, stride=STRIDE, truncation=True, return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding='max_length', return_tensors='pt')
        ids, am = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        probs = {k: [] for k in M.HEADS}
        with torch.no_grad():
            for b in range(0, ids.shape[0], BATCH):
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    out = self.model(ids[b:b + BATCH], am[b:b + BATCH])
                for k in M.HEADS:
                    probs[k].append(torch.softmax(out[f'{k}_logits'].float(), -1).cpu())
        probs = {k: torch.cat(v) for k, v in probs.items()}
        char = {}  # ci -> dict head -> prob vector (first window wins)
        for w in range(ids.shape[0]):
            for ti, (cs, ce) in enumerate(enc['offset_mapping'][w].tolist()):
                if cs == ce or am[w, ti] == 0:
                    continue
                for ci in range(cs, ce):
                    if ci not in char:
                        char[ci] = {k: probs[k][w, ti] for k in M.HEADS}
                        char[ci]['_tok'] = (w, ti)
        return char

    def predict(self, text):
        char = self._char_probs(text)
        spans, cur = [], None
        for ci in range(len(text)):
            lab = M.SPAN_LABELS[int(char[ci]['span'].argmax())] if ci in char else 'O'
            new_token = ci in char and (ci - 1 not in char or char[ci - 1]['_tok'] != char[ci]['_tok'])
            if (lab == 'B' and new_token) or (lab in ('B', 'I') and cur is None):
                if cur:
                    spans.append(cur)
                cur = [ci, ci + 1]
            elif lab in ('B', 'I'):
                cur[1] = ci + 1
            elif cur and text[ci].isspace() and ci + 1 < len(text) and ci + 1 in char \
                    and M.SPAN_LABELS[int(char[ci + 1]['span'].argmax())] == 'I':
                cur[1] = ci + 1  # whitespace inside a multi-word value ("hand movements")
            else:
                if cur:
                    spans.append(cur); cur = None
        if cur:
            spans.append(cur)
        out = []
        for s, e in spans:
            while e > s and text[e - 1].isspace():
                e -= 1
            cs = [char[c] for c in range(s, e) if c in char and not text[c].isspace()]
            if not cs:
                continue
            r = {'s': s, 'e': e, 'val': text[s:e], 'span_conf': float(sum(1 - c['span'][0] for c in cs) / len(cs))}
            for f in FIELDS:
                p = sum(c[f] for c in cs) / len(cs)
                labels = M.HEADS[f]
                i = int(p.argmax())
                r[f] = labels[i]; r[f + '_conf'] = float(p[i]); r[f + '_dist'] = [round(float(x), 4) for x in p]
            out.append(r)
        return out


SNELLEN_M_DEN = {'3', '3.8', '4', '4.8', '5', '6', '7.5', '9', '9.5', '12', '15', '18', '19', '24', '30', '36', '38', '48', '60', '75', '95', '120'}
SNELLEN_M_NUM = {'1', '2', '3', '4', '5', '6'}
QUAL_RE = re.compile(r"^(?:CF|HM|PL|NPL|NLP|PL\s*\+|(?:no\s+)?(?:perception\s+of\s+light|light\s+perception)(?:\s+only)?"
                     r"|hand\s+movements?(?:\s+only)?|count(?:ing|ed)?\s+fingers?(?:\s+at\s+\S+)?)$", re.I)


def value_plausible(val, notation):
    """Grammar check of the value against its notation. Only rejects what
    can't be a reading in that notation; it never changes a value."""
    v = re.sub(r"\s*[+-]\s*\d$", "", val.strip())  # Snellen partial-line suffix "6/12-2"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", v)
    try:
        if notation == 'snellen_m':
            return bool(m) and m.group(1) in SNELLEN_M_NUM and m.group(2) in SNELLEN_M_DEN
        if notation == 'snellen_ft':
            return bool(m) and m.group(1) == '20' and 10 <= float(m.group(2)) <= 800
        if notation == 'qualitative':
            return bool(QUAL_RE.match(val.strip()))
        if notation == 'etdrs':
            return re.fullmatch(r"\d{1,3}", v) is not None and 0 <= int(v) <= 100
        if notation == 'logmar':
            return -0.3 <= float(v) <= 3.0
        if notation in ('decimal', 'thomson'):
            return 0.0 < float(v) <= 2.0
        if notation == 'Unk':  # bare decimal, chart not named: plausible as decimal or LogMAR
            return -0.3 <= float(v) <= 3.0
    except ValueError:
        return False
    return False


def gate(text, spans, tau, evidence=None):
    """spans: output of VAModel.predict (or any list with s, e, lat/method/
    notation and *_conf). tau: dict field -> confidence threshold.
    Adds per field: <f>_final (value, 'Unk' or 'Review'), <f>_state
    ('committed' | 'unk' | 'review'), <f>_reason."""
    ev = evidence if evidence is not None else label_document(text, [(x['s'], x['e']) for x in spans])
    for x, r in zip(spans, ev):
        for f in FIELDS:
            mv, conf, rv = x[f], x[f + '_conf'], r[f]
            agree = mv == rv
            if conf < tau[f]:
                x[f + '_final'], x[f + '_state'], x[f + '_reason'] = 'Review', 'review', 'low_confidence'
            elif agree and mv == 'Unk':
                x[f + '_final'], x[f + '_state'], x[f + '_reason'] = 'Unk', 'unk', 'not_stated'
            elif agree:
                x[f + '_final'], x[f + '_state'], x[f + '_reason'] = mv, 'committed', 'model_and_evidence_agree'
            else:
                x[f + '_final'], x[f + '_state'] = 'Review', 'review'
                x[f + '_reason'] = f"model={mv} evidence={rv}"
            x[f + '_evidence'] = r.get(f + '_ev')
        notation = x['notation_final'] if x['notation_final'] not in ('Review',) else x['notation']
        ok = value_plausible(x['val'], notation)
        x['value_state'] = 'committed' if ok else 'review'
        x['value_reason'] = None if ok else f"value not plausible as {notation}"
    return spans
