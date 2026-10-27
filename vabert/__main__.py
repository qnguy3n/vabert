"""
Command-line interface of the vabert package. Run as
`vabert ...` after `pip install -e <release folder>`, or as
`python -m vabert ...` from the release folder.

Single letter:
    vabert --text "Visual acuity unaided was 6/9 in the right eye."
    vabert --file letter.txt [--out results.csv] [--json]
    cat letter.txt | vabert --stdin

Many letters, one output row per detected reading:
    vabert --corpus letters.csv --text-column letter_text [--id-column id] --out results.csv

Each row has the reading, its final laterality / method / notation (a
value, 'Unk' when the letter doesn't state it, or 'Review'), the state
and reason for each field, the model's own answer and confidence, the
evidence text the rule check found, and the letter's overall status.
No network access: everything runs from the model/ checkpoint (override
with --model-dir or VABERT_MODEL_DIR).
"""

import argparse, csv, json, sys

from .pipeline import VABERTPipeline
from .inference import FIELDS

COLUMNS = (['letter_status', 'label', 'va_value', 'start', 'end', 'value_state', 'value_reason', 'span_confidence']
           + [f'{f}{s}' for f in FIELDS for s in ('', '_state', '_reason', '_model', '_confidence', '_evidence')])


def _rows(result):
    return [{'letter_status': result['status'], **r} for r in result['readings']]


def _print(rows, status):
    print(f"Letter status: {status}")
    if not rows:
        print("No VA readings detected.")
        return
    for r in rows:
        print(f"  {r['va_value']:>10}  {r['label']:<24} notation={r['notation']:<11} "
              f"value={r['value_state']}  lat={r['lat_state']}  method={r['method_state']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vabert', description='Extract visual acuity readings (value, eye, method, notation) from clinic letter text.')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--text', help='letter text')
    g.add_argument('--file', help='a text file holding one letter')
    g.add_argument('--stdin', action='store_true', help='read one letter from standard input')
    g.add_argument('--corpus', help='a CSV file with one letter per row (use with --out)')
    ap.add_argument('--text-column', default='text', help='--corpus: column holding the letter text (default: text)')
    ap.add_argument('--id-column', help='--corpus: column holding a letter id, copied to the output')
    ap.add_argument('--out', help='write a CSV of readings here'); ap.add_argument('--json', action='store_true', help='print the full result as JSON')
    ap.add_argument('--model-dir', help='checkpoint folder (default: model/ in the release folder)')
    ap.add_argument('--device', help="e.g. 'cpu' or 'cuda:0' (default: GPU if available)")
    a = ap.parse_args(argv)
    pipe = VABERTPipeline(model_dir=a.model_dir, device=a.device)
    if a.corpus:
        csv.field_size_limit(sys.maxsize)
        rows = []
        with open(a.corpus, newline='') as f:
            for i, rec in enumerate(csv.DictReader(f)):
                res = pipe.run(rec[a.text_column] or '')
                rid = rec[a.id_column] if a.id_column else i
                rows += [{'id': rid, **r} for r in _rows(res)] or [{'id': rid, 'letter_status': res['status']}]
                if (i + 1) % 50 == 0:
                    print(f"{i + 1} letters", file=sys.stderr)
        cols = ['id'] + COLUMNS
    else:
        text = a.text if a.text is not None else open(a.file).read() if a.file else sys.stdin.read()
        res = pipe.run(text)
        rows, cols = _rows(res), COLUMNS
        if a.json:
            print(json.dumps(res, indent=2))
        else:
            _print(rows, res['status'])
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {a.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
