"""
postprocess.py — suppression filters applied to the model's spans before
the commit gate. They only remove things that aren't VA readings; they
never assign an eye or method.

is_plausible_va() drops Snellen-shaped spans whose numerator isn't a real
testing distance AND which have a corroborating non-VA signal nearby
(blood pressure, a follow-up interval such as "review in 6/12").

is_fragment() drops a span whose value can't be a VA reading on its own: a
lone letter or two that isn't a qualitative code ("N", "F" cut out of a
word) or a fraction with an empty side ("/", "1/", "/24").
"""

import re

# ── 2. Plausibility filter ──────────────────────────────────────────────────
# Real Snellen numerators are drawn from a small set of standard testing
# distances (6/20 metres/feet dominate; 1-5, 9, 12 appear in near-vision and
# specialised charts). A blood-pressure or other clinical-ratio numerator
# (e.g. 154 in "154/96mmHg") never falls in this set.
SNELLEN_VALID_NUMERATORS = {'1', '2', '3', '4', '5', '6', '9', '12', '20'}
FRACTION_RE = re.compile(r'^(\d+)[ \t]{0,3}/[ \t]{0,3}(\d+)')
MMHG_AFTER_RE = re.compile(r'^\s{0,3}mmhg', re.IGNORECASE)
BP_CONTEXT_RE = re.compile(r'\bblood pressure\b|\bsystolic\b|\bdiastolic\b|\bBP\b', re.IGNORECASE)
BP_CONTEXT_WINDOW = 40

# Round 2: "review in 6/12", "F/U 3/12", "repeat 6/52" -- common UK clinical
# shorthand for review-in-N-months/weeks, colliding with real Snellen
# denominators (6/12 is also an entirely ordinary VA reading). Only catches
# instances with an explicit nearby keyword -- see module docstring, item 6.
FOLLOWUP_CONTEXT_RE = re.compile(
    r'\breview\b|\bfollow[\s-]?up\b|\bf/u\b|\brepeat\b|\bmonths?\b|\bweeks?\b|\bclinic\b',
    re.IGNORECASE)


def is_plausible_va(text, start, end, value):
    """Returns False only for spans that are structurally implausible AND
    have a corroborating non-VA signal nearby (blood pressure, or a
    follow-up-interval keyword) -- conservative by design, so an
    unusual-but-real reading is never dropped on numerator shape alone.
    Non-fraction values (qualitative, decimal, ETDRS, LogMAR) are
    untouched by this filter."""
    m = FRACTION_RE.match(value)
    if not m:
        return True
    if m.group(1) in SNELLEN_VALID_NUMERATORS:
        return True
    after = text[end:end + 8]
    before_ctx = text[max(0, start - BP_CONTEXT_WINDOW):start]
    after_ctx = text[end:end + BP_CONTEXT_WINDOW]
    if MMHG_AFTER_RE.match(after) or BP_CONTEXT_RE.search(before_ctx) or BP_CONTEXT_RE.search(after_ctx):
        return False
    if FOLLOWUP_CONTEXT_RE.search(before_ctx) or FOLLOWUP_CONTEXT_RE.search(after_ctx):
        return False
    return True


QUALITATIVE_CODES = {'CF', 'HM', 'PL', 'LP', 'NPL', 'NLP'}
LETTERS_RE = re.compile(r'[A-Za-z]{1,2}')
PARTIAL_FRACTION_RE = re.compile(r'\d*\s*/\s*\d*')
FULL_FRACTION_RE = re.compile(r'\d+\s*/\s*\d+')


def is_fragment(value):
    v = value.strip()
    if LETTERS_RE.fullmatch(v):
        return v.upper() not in QUALITATIVE_CODES
    return bool(PARTIAL_FRACTION_RE.fullmatch(v)) and not FULL_FRACTION_RE.fullmatch(v)


def filter_implausible(text, spans):
    return [x for x in spans if not is_fragment(x['val']) and is_plausible_va(text, x['s'], x['e'], x['val'])]
