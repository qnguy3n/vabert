"""
evidence_rules.py — rule-based evidence labeller for the explicit label
scheme.

Given a document and the character spans of its VA values, decides for
each span what the TEXT states about it, never what is usual or likely:

  laterality: Right | Left | Both | Unk
  method:     Aided | Unaided | AidedPH | UnaidedPH | UnkPH | Unk
  notation:   snellen_m | snellen_ft | qualitative | etdrs | logmar |
              decimal | thomson | Unk

and returns the character offsets of the evidence it used. The same rules
labelled the training data, and the pipeline uses them as the evidence
check: a field is only committed when the model and this labeller agree.

Scope rules:
  Tables (vertical EPR layout, flattened one-line layout, markdown):
    laterality comes from the column header (Right Eye / Left Eye),
    method from the row label (Unaided / Glasses / Pinhole ...).
    A table with no row label gives method Unk.
  Narrative: the segment containing the span (a line, split further at
    sentence ends and semicolons).
    Laterality: if the segment starts with an eye word before its first
      value, each value takes the nearest eye word before it ("RE 6/9,
      LE 6/6"); otherwise each value takes the nearest eye word after it
      ("6/9 on the right and 6/6 on the left"). An eye word more than
      MAX_LAT_GAP characters away doesn't count.
    Method: a method phrase directly after a value (only whitespace
      between, e.g. "6/18ph", "6/15 pinhole") applies to that value only.
      Any other method phrase applies to every later value in the
      segment until the next method phrase. A heading line ending in ":"
      that states a method carries into the following non-blank lines.
  Pinhole stated without aided/unaided gives UnkPH. No eye word gives
  Unk, with no first-is-right convention. "both eyes"/BE/OU gives Both.
  A bare decimal with no chart named in scope gives notation Unk; a
  negative value is LogMAR.
"""

import re
from bisect import bisect_right

LATERALITIES = ['Right', 'Left', 'Both', 'Unk']
METHODS = ['Aided', 'Unaided', 'AidedPH', 'UnaidedPH', 'UnkPH', 'Unk']
NOTATIONS = ['snellen_m', 'snellen_ft', 'qualitative', 'etdrs', 'logmar', 'decimal', 'thomson', 'Unk']

OLD_EYE = {'OD': 'Right', 'OS': 'Left'}
OLD_METHOD = {'CC': 'Aided', 'SC': 'Unaided', 'CCPH': 'AidedPH', 'SCPH': 'UnaidedPH'}


def old_tag_to_new(tag):
    """VAODDISTCC -> ('Right', 'Aided')."""
    m = re.match(r'^VA(OD|OS)DIST(CCPH|SCPH|CC|SC)$', tag)
    return OLD_EYE[m.group(1)], OLD_METHOD[m.group(2)]


def make_label(lat, method):
    return f"VA-{lat}-{method}"


def split_label(label):
    _, lat, method = label.split('-')
    return lat, method


# ── vocabularies ────────────────────────────────────────────────────────────
MAX_LAT_GAP = 80
MAX_SUFFIX_GAP = 30

LAT_RE = re.compile(
    r"(?P<Both>\bboth\s+eyes\b|\bBE\b|\bOU\b|\bB/E\b|\beach\s+eye\b|\bbilaterally\b)"
    r"|(?P<Right>\bright(?:\s+eye)?\b|\bRE\b|\bR\.E\.|\bRVA\b|\bOD\b|\bRT\b|\bR\b(?!\s*&)(?![/'’]))"
    r"|(?P<Left>\bleft(?:\s+eye)?\b|\bLE\b|\bL\.E\.|\bLVA\b|\bOS\b|\bLT\b|\bL\b(?!\s*&)(?![/'’]))",
    re.I)
# single letters and two-letter codes only count in their usual case
_CASE_SENSITIVE = {'R', 'L', 'RE', 'LE', 'OD', 'OS', 'OU', 'BE', 'RT', 'LT', 'RVA', 'LVA'}

# phrases that mention pinhole without meaning "this value was through a pinhole"
PH_NEGATIVE_RE = re.compile(
    r"\bNIPH\b|\bPHNI\b|\bno\s+(?:improvement|increase|change|better|gain)\s+(?:with|on|through|using)\s+(?:a\s+)?(?:pinhole|PH)\b"
    r"|\b(?:does|did)\s+not\s+(?:improve\s+(?:with|on)\s+(?:a\s+)?)?(?:pinhole|PH)\b|\bnot\s+improv\w*\s+(?:with|on)\s+(?:a\s+)?(?:pinhole|PH)\b", re.I)
# anything that looks like a VA value, labelled or not; used only to decide
# which value a method word sits next to
# a value-shaped token ending right before a method word ("6/18ph"), for
# method words that belong to a value which isn't one of the spans
VALUE_BEFORE_RE = re.compile(r"(?<![\w/.])(?:\d{1,2}/\d{1,3}(?:\.\d+)?|CF|HM|PL|NPL|NLP)[ \t\xa0]{0,2}$")
VALUE_TOKEN_RE = re.compile(r"(?<![\w/.])(?:\d{1,2}/\d{1,3}(?:\.\d+)?|-?[0-2]\.\d{1,3}|CF|HM|PL|NPL|NLP)(?![\w/])")
METHOD_RE = re.compile(
    r"(?P<Unaided>\bunaided\b|\buncorrected\b|\bwithout\s+(?:glasses|spectacles|specs|correction|(?:contact\s+)?lens(?:es)?)\b|\bsc\b|\bUA\b|\bnaked\s+eye\b)"
    r"|(?P<Aided>\baided\b|\bbest[\s-]+corrected\b|\bBCVA\b|\bcorrected\b|\bwith\s+(?:(?:his|her|their|own)\s+)?(?:glasses|spectacles|specs|correction|contact\s+lens(?:es)?|CLs?)\b"
    r"|\bwearing\s+(?:(?:his|her|their)\s+)?(?:glasses|spectacles|specs|contact\s+lens(?:es)?)\b|\b(?:own\s+)?(?:glasses|spectacles|specs)\b|\bcontact\s+lens(?:es)?\b|\bCLs?\b|\bcc\b)"
    r"|(?P<Pinhole>\b(?:with|through|on|using)\s+(?:a\s+)?(?:pinhole|PH)\b|\bpinhole\b|\bPH\b|(?<=\d)ph\b|\bph\b)",
    re.I)
_METHOD_CASE_SENSITIVE = {'sc', 'cc', 'UA', 'CL', 'CLs', 'PH', 'BCVA'}

CHART_RE = re.compile(
    r"(?P<logmar>\blog\s*-?\s*mar\b)|(?P<thomson>\bthomson\b)|(?P<decimal>\bdecimal\b)|(?P<etdrs>\betdrs\b|\bletters\b)",
    re.I)

TABLE_METHOD_LABELS = [
    (r"unaided\s+pinhole|pinhole\s+unaided", 'UnaidedPH'),
    (r"(?:glasses|spectacles|aided)\s+(?:\+\s*|with\s+)?pinhole|pinhole\s+(?:with|over|\+)\s+(?:glasses|spectacles)", 'AidedPH'),
    (r"unaided|uncorrected|without\s+(?:glasses|spectacles|correction)", 'Unaided'),
    (r"glasses|spectacles|specs|aided|corrected|contact\s+lens(?:es)?|cl|with\s+(?:glasses|spectacles|correction)", 'Aided'),
    (r"pinhole|ph", 'UnkPH'),
    (r"habitual", 'Unk'),
]
TABLE_METHOD_RE = re.compile(r"(?:" + "|".join(p for p, _ in TABLE_METHOD_LABELS) + r")(?=\s|$|\|)", re.I)

VALUE_CELL_RE = re.compile(
    r"(?:-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?:\s*[+-]\s*\d)?(?:\s*\(\s*\d+/\d+(?:\.\d+)?\s*\))?"
    r"|NIPH|PHNI|N/?A|-|–"
    r"|(?:no\s+)?(?:perception\s+of\s+light|light\s+perception)(?:\s+only)?|hand\s+movements?(?:\s+only)?"
    r"|counting\s+fingers|count\s+fingers|CF(?:\s+at\s+\d+\s*(?:m|cm|ft))?|HM|PL|NPL|NLP)(?=\s|$|\|)",
    re.I)
UNKNOWN_ROW_RE = re.compile(r"[A-Za-z][A-Za-z()\-]*(?:[ \t]+[A-Za-z()\-]+){0,2}[ \t]*(?=\r?\n)")
HEADER_RE = re.compile(r"\b(right|left)(?:\s+eye)?\b[ \t]*(?:\n[ \t]*|\|[ \t]*|[ \t]+)\b(right|left)(?:\s+eye)?\b", re.I)


def _lat_value(m):
    for k in ('Both', 'Right', 'Left'):
        if m.group(k):
            return k
    return None


def _ok_case(tok, case_set):
    return tok not in {t.lower() for t in case_set} or tok in case_set or tok.lower() in {'right', 'left', 'both eyes', 'each eye'}


def find_lat_hits(text, lo, hi):
    out = []
    for m in LAT_RE.finditer(text, lo, hi):
        tok = m.group(0)
        if tok.upper() in _CASE_SENSITIVE and tok not in _CASE_SENSITIVE:
            continue
        out.append((m.start(), m.end(), _lat_value(m)))
    return out


def find_method_hits(text, lo, hi):
    masked = [(m.start(), m.end()) for m in PH_NEGATIVE_RE.finditer(text, lo, hi)]
    out = []
    for m in METHOD_RE.finditer(text, lo, hi):
        tok = m.group(0)
        if any(a <= m.start() < b for a, b in masked):
            continue
        if tok.lower() in {t.lower() for t in _METHOD_CASE_SENSITIVE} and tok not in _METHOD_CASE_SENSITIVE:
            if not (m.group('Pinhole') and tok.lower() == 'ph'):
                continue
        kind = 'Unaided' if m.group('Unaided') else 'Aided' if m.group('Aided') else 'Pinhole'
        out.append((m.start(), m.end(), kind))
    return out


def combine_methods(kinds):
    """Set of {'Aided','Unaided','Pinhole'} stated together -> method value."""
    base = 'Aided' if 'Aided' in kinds else 'Unaided' if 'Unaided' in kinds else None
    if 'Aided' in kinds and 'Unaided' in kinds:
        return None  # contradictory in one phrase
    if 'Pinhole' in kinds:
        return {'Aided': 'AidedPH', 'Unaided': 'UnaidedPH', None: 'UnkPH'}[base]
    return base or 'Unk'


# ── tables ──────────────────────────────────────────────────────────────────
def _table_method(label):
    lab = label.strip().lower()
    for pat, val in TABLE_METHOD_LABELS:
        if re.fullmatch(pat, lab, re.I):
            return val
    return None


def parse_tables(text):
    """Returns list of cells: dict(s, e, lat, lat_ev, method, method_ev, header_lo).
    Covers the vertical EPR layout, its one-line flattened form, and
    markdown pipe tables."""
    cells = []
    for hm in HEADER_RE.finditer(text):
        cols = [hm.group(1).capitalize(), hm.group(2).capitalize()]
        if cols[0] == cols[1]:
            continue
        col_ev = [(hm.start(1), hm.end(1)), (hm.start(2), hm.end(2))]
        line_start = text.rfind('\n', 0, hm.start()) + 1
        header_lo = text.rfind('\n', 0, max(0, line_start - 1)) + 1  # include the line above (e.g. "Visual Acuity (LogMAR)")
        header_line = text[line_start:text.find('\n', hm.end()) if text.find('\n', hm.end()) >= 0 else len(text)]
        if '|' in header_line:
            cells += _parse_markdown(text, hm, cols, col_ev, header_lo)
        else:
            cells += _parse_stream(text, hm.end(), cols, col_ev, header_lo)
    return cells


def _parse_stream(text, pos, cols, col_ev, header_lo):
    """Vertical or flattened layout: tokens separated by whitespace/newlines.
    A row label starts a row; following value cells fill the columns."""
    out, method, method_ev, col = [], 'Unk', None, 0
    n = len(text)
    blank_run = 0
    while pos < n:
        ws = re.match(r"[ \t\r\n\xa0]*", text[pos:]).group(0)
        blank_run = blank_run + ws.count('\n') if ws.strip(' \t\r\xa0') == ws.strip(' \t\r\xa0') else 0
        pos += len(ws)
        if pos >= n or ws.count('\n') > 6:
            break
        rest = text[pos:]
        lm = TABLE_METHOD_RE.match(rest)
        vm = VALUE_CELL_RE.match(rest)
        if lm and (not vm or lm.end() >= vm.end()):
            method = _table_method(lm.group(0)) or 'Unk'
            method_ev = (pos, pos + lm.end())
            col = 0
            pos += lm.end()
            continue
        um = UNKNOWN_ROW_RE.match(rest) if '\n' in ws else None
        if um and not vm:
            nxt = text[pos + um.end():].lstrip(' \t\r\n\xa0')
            if VALUE_CELL_RE.match(nxt):
                method, method_ev, col = 'Unk', None, 0  # e.g. "Formal refraction": a row we can't map to a method
                pos += um.end()
                continue
        if vm and vm.end() > 0:
            if col < len(cols):
                out.append(dict(s=pos, e=pos + vm.end(), lat=cols[col], lat_ev=col_ev[col],
                                method=method, method_ev=method_ev, header_lo=header_lo))
            col += 1
            pos += vm.end()
            continue
        break
    return out


def _parse_markdown(text, hm, cols, col_ev, header_lo):
    out = []
    line_end = text.find('\n', hm.end())
    pos = line_end + 1 if line_end >= 0 else len(text)
    hdr = text[text.rfind('\n', 0, hm.start()) + 1:line_end if line_end >= 0 else len(text)]
    n_lead = len(hdr.split('|')) - 2  # cells before the two eye columns (row-label column)
    while pos < len(text):
        le = text.find('\n', pos)
        le = len(text) if le < 0 else le
        line = text[pos:le]
        if '|' not in line:
            break
        if re.fullmatch(r"[\s|:-]*", line):
            pos = le + 1
            continue
        parts, offs, p = line.split('|'), [], pos
        for part in parts:
            offs.append(p); p += len(part) + 1
        label = parts[0] if n_lead >= 1 else ''
        method = (_table_method(label) or 'Unk') if label.strip() else 'Unk'
        method_ev = (offs[0], offs[0] + len(parts[0])) if method != 'Unk' else None
        for ci in range(2):
            k = ci + n_lead
            if k < len(parts):
                cs = offs[k] + (len(parts[k]) - len(parts[k].lstrip()))
                ce = offs[k] + len(parts[k].rstrip())
                if ce > cs:
                    out.append(dict(s=cs, e=ce, lat=cols[ci], lat_ev=col_ev[ci], method=method,
                                    method_ev=method_ev, header_lo=header_lo))
        pos = le + 1
    return out


# ── narrative ───────────────────────────────────────────────────────────────
SENT_SPLIT_RE = re.compile(r"(?<=[A-Za-z0-9\)\]])\.+\s+(?=(?:[-–•]\s*)?[A-Z])|;|\n")


def segment_bounds(text, s):
    """[lo, hi) of the segment containing position s."""
    lo, hi = 0, len(text)
    for m in SENT_SPLIT_RE.finditer(text):
        if m.end() <= s:
            lo = m.end()
        elif m.start() >= s:
            hi = m.start()
            break
    return lo, hi


def _heading_method(text, lo):
    """If the segment is preceded by a heading line ending in ':' that
    states a method (and nothing blank in between), return its hits."""
    prev_end = lo - 1
    while prev_end > 0 and text[prev_end - 1] != '\n':
        if text[prev_end - 1] not in ' \t\xa0':
            break
        prev_end -= 1
    p = lo
    while True:
        ls = text.rfind('\n', 0, max(0, p - 1)) + 1
        line = text[ls:p].strip()
        if not line or ls == 0 and not line:
            return []
        if line.endswith(':'):
            hits = find_method_hits(text, ls, p)
            return hits
        if VALUE_CELL_RE.search(line) is None:
            return []
        p = ls
        if ls == 0:
            return []


# eye stated right next to a value. Suffix forms: "6/9 on the right",
# "CF NIPH on the left eye", "6/12 left." (a bare eye word counts as a
# suffix only when no value follows it, else it labels the next value:
# "Right 6/6 left 6/9"). Prefix forms: "RE 6/9", "Right: 6/6", "left eye 6/24".
SUFFIX_PREP_RE = re.compile(r"(?:[ \t]*\([^()\n]{1,15}\))?[ \t]*(?:(?:NIPH|PHNI|ph|PH|pinhole|unaided|aided|corrected)[ \t,]*)?(?:,[ \t]*)?(?:on|in|for|at)[ \t]+(?:the[ \t]+)?(?:(?:his|her|their|patient's)[ \t]+)?(right|left)\b(?:[ \t]+eye)?", re.I)
SUFFIX_BOTH_RE = re.compile(r"(?:[ \t]*\([^()\n]{1,15}\))?[ \t]*(?:(?:on|in|for|at)[ \t]+)?(?:both[ \t]+eyes|each[ \t]+eye|BE\b|OU\b|bilaterally\b)")
SUFFIX_BARE_RE = re.compile(r"(?:[ \t]*\([^()\n]{1,15}\))?[ \t]*(?:(?:NIPH|PHNI)[ \t]*)?\(?(right|left|RE|LE|OD|OS|R|L)\b\)?(?:[ \t]+eye)?(?=[ \t]*(?:[,.;)]|and\b|$|\n))")
PREFIX_RE = re.compile(r"\b(right|left|RE|LE|OD|OS|R|L|RVA|LVA)\b(?:[ \t]+eye)?[ \t]*[:=\-)]?[ \t]*$")


def _lat_word(w):
    w2 = w.lower()
    if w2 in ('right', 're', 'od', 'r', 'rva'):
        return 'Right'
    if w2 in ('left', 'le', 'os', 'l', 'lva'):
        return 'Left'
    return None


def _case_ok(w):
    return w.lower() in ('right', 'left') or w == w.upper()


def _local_laterality(text, s, e, nxt_s, seg_lo, prev_e=None):
    """(lat, (start, end)) if an eye is stated immediately around the value,
    'Unk'-free: returns None when there is no strong local evidence, and
    ('Unk', None) when prefix and suffix name different eyes."""
    after = text[e:e + 60]
    lim = nxt_s - e
    cands = []
    m = SUFFIX_PREP_RE.match(after)
    if m and m.end() > lim:
        m = None
    if m:
        cands.append((_lat_word(m.group(1)), (e + m.start(1), e + m.end(1)), 'suffix'))
    else:
        m = SUFFIX_BOTH_RE.match(after)
        if m and m.end() > lim:
            m = None
        if m and m.end() > len(m.group(0)) - len(m.group(0).lstrip()):
            cands.append(('Both', (e + m.start(), e + m.end()), 'suffix'))
        else:
            m = SUFFIX_BARE_RE.match(after)
            if m and m.end() > lim:
                m = None
            if m and _case_ok(m.group(1)):
                cands.append((_lat_word(m.group(1)), (e + m.start(1), e + m.end(1)), 'suffix'))
    before = text[max(seg_lo, s - 25):s]
    m = PREFIX_RE.search(before)
    if m and _case_ok(m.group(1)):
        off = s - len(before)
        ws = off + m.start(1)
        # an eye word sitting right after another value may be that value's
        # suffix ("no light perception right hand movements left"): not local
        # evidence for this one either way
        if not (prev_e is not None and re.fullmatch(r"[ \t\xa0]{0,2}", text[prev_e:ws])):
            cands.append((_lat_word(m.group(1)), (ws, off + m.end(1)), 'prefix'))
    if not cands:
        return None
    if len({c[0] for c in cands}) == 1:
        return cands[0][0], cands[0][1]
    return 'Unk', None


BRACKET_GAP_RE = re.compile(r"[ \t]*\([ \t]*")


def narrative_attributes(text, spans):
    """spans: list of (s, e). Returns list of dicts lat/lat_ev/method/method_ev.
    A value written in brackets right after another value ("0.40 (6/15)")
    is the same reading in a second notation: it takes the attributes of
    the value it follows and is otherwise ignored."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    host = {}
    for a, b in zip(order, order[1:]):
        if BRACKET_GAP_RE.fullmatch(text[spans[a][1]:spans[b][0]]) and text[spans[b][1]:spans[b][1] + 3].lstrip().startswith(')'):
            host[b] = host.get(a, a)
    primary = [i for i in range(len(spans)) if i not in host]
    sub = _narrative_attributes(text, [spans[i] for i in primary])
    res = [None] * len(spans)
    for i, r in zip(primary, sub):
        res[i] = r
    for b, a in host.items():
        res[b] = dict(res[a])
    return res


def _narrative_attributes(text, spans):
    res = [None] * len(spans)
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    by_seg = {}
    for i in order:
        by_seg.setdefault(segment_bounds(text, spans[i][0]), []).append(i)
    for (lo, hi), idxs in by_seg.items():
        lat_hits = [h for h in find_lat_hits(text, lo, hi)
                    if not any(spans[j][0] <= h[0] < spans[j][1] for j in idxs)]
        meth_hits = [h for h in find_method_hits(text, lo, hi)
                     if not any(spans[j][0] <= h[0] < spans[j][1] for j in idxs)]
        first_val = min(spans[j][0] for j in idxs)
        prefix_mode = any(h[1] <= first_val for h in lat_hits)
        # laterality: strong local evidence first ...
        for j in idxs:
            s, e = spans[j]
            nxt_s = min([spans[k][0] for k in idxs if spans[k][0] >= e and k != j] or [hi])
            prev_e = max([spans[k][1] for k in idxs if spans[k][1] <= s and k != j] or [None], key=lambda v: -1 if v is None else v)
            for m in VALUE_TOKEN_RE.finditer(text, max(lo, s - 40), s):
                prev_e = m.end() if prev_e is None or m.end() > prev_e else prev_e
            local = _local_laterality(text, s, e, nxt_s, lo, prev_e)
            if local is not None:
                res[j] = {'lat': local[0], 'lat_ev': local[1]}
                continue
            # ... then the segment's prefix/suffix pattern
            if prefix_mode:
                cands = [h for h in lat_hits if h[1] <= s]
                h = max(cands, key=lambda h: h[1]) if cands else None
                # carrying an eye word past another value is allowed ("RE 6/36 ph 6/9")
                # unless a new "label:" starts in between ("L: 6/7.5, PH: 6/6")
                if h:
                    between = [spans[k] for k in idxs if h[1] <= spans[k][0] and spans[k][1] <= s]
                    if between and ':' in text[max(b for _, b in between):s]:
                        h = None
                gap = s - h[1] if h else None
            else:
                cands = [h for h in lat_hits if h[0] >= e]
                h = min(cands, key=lambda h: h[0]) if cands else None
                # the eye word must come before the next value
                if h:
                    nxt = [spans[k][0] for k in idxs if spans[k][0] >= e and k != j]
                    if nxt and min(nxt) < h[0]:
                        h = None
                if h and (h[0] - e > MAX_SUFFIX_GAP or VALUE_TOKEN_RE.search(text, e, h[0])):
                    h = None
                gap = h[0] - e if h else None
            if h and gap <= MAX_LAT_GAP:
                lat, lat_ev = h[2], (h[0], h[1])
            else:
                lat, lat_ev = 'Unk', None
            res[j] = {'lat': lat, 'lat_ev': lat_ev}
        # method: merge adjacent hits into phrases first
        phrases = []
        for h in sorted(meth_hits):
            if phrases and h[0] - phrases[-1]['e'] <= 12 and not any(phrases[-1]['e'] <= spans[k][0] < h[0] for k in idxs) \
                    and re.fullmatch(r"[\s,+&/-]*(?:and|with|through|then|over|on)?[\s,+&/-]*(?:a\s+)?", text[phrases[-1]['e']:h[0]], re.I):
                phrases[-1]['e'] = h[1]; phrases[-1]['kinds'].add(h[2])
            else:
                phrases.append({'s': h[0], 'e': h[1], 'kinds': {h[2]}})
        heading = _heading_method(text, lo)
        current = None
        if heading:
            kinds = {h[2] for h in heading}
            current = {'s': heading[0][0], 'e': heading[-1][1], 'kinds': kinds}
        events = sorted([(p['s'], 'P', p) for p in phrases] + [(spans[j][0], 'V', j) for j in idxs], key=lambda x: (x[0], x[1] == 'V'))
        # Each phrase is one of: a suffix of the value right before it
        # ("6/18ph", "6/15 pinhole"; whitespace only in between), a prefix
        # that opens a scope for later values ("unaided: RE 6/9, LE 6/6"),
        # or a tail of the value before it ("6/6 left with glasses"). A
        # tail never opens a scope; if it disagrees with the scope already
        # in force, the value gets Unk and a conflict flag.
        role = {}
        for p in phrases:
            prev = [k for k in idxs if spans[k][1] <= p['s']]
            nxt = [k for k in idxs if spans[k][0] >= p['e']]
            k = max(prev, key=lambda k: spans[k][1]) if prev else None
            n = min(nxt, key=lambda k: spans[k][0]) if nxt else None
            after_prev = k is not None and re.fullmatch(r"[ \t\xa0]{0,2}", text[spans[k][1]:p['s']])
            nv = VALUE_TOKEN_RE.match(text, p['e'] + len(re.match(r"[ \t\xa0]{1,2}", text[p['e']:]).group(0))) \
                if re.match(r"[ \t\xa0]{1,2}", text[p['e']:]) else None
            before_next = (n is not None and re.fullmatch(r"[ \t\xa0]{1,2}", text[p['e']:spans[n][0]])) or nv is not None
            if not after_prev and VALUE_BEFORE_RE.search(text[max(0, p['s'] - 12):p['s']]):
                if before_next and n is not None:
                    role[id(p)] = ('ambiguous', (n,))  # "6/18 pH 6/12" with 6/18 not a span
                else:
                    role[id(p)] = ('orphan', None)  # "RE 6/60, 6/18ph" with 6/18 not a span: not about 6/60
            elif re.match(r"[ \t\xa0]*:", text[p['e']:]):
                if after_prev and n is not None:
                    role[id(p)] = ('label_next', n)  # "RE: 6/48 PH: 6/15": labels 6/15 only
                else:
                    role[id(p)] = ('prefix', None)  # "Vision (unaided): R 6/6, L 6/7.5" opens a scope
            elif after_prev and before_next:
                role[id(p)] = ('ambiguous', (k, n))  # "6/36 ph 6/9": suffix of 6/36 or prefix of 6/9?
            elif after_prev:
                role[id(p)] = ('suffix', k)
            elif k is not None and p['s'] - spans[k][1] <= 30 and (n is None or len(text[p['e']:spans[n][0]].strip(' :-\t')) > 3):
                role[id(p)] = ('tail', k)
            else:
                role[id(p)] = ('prefix', None)
        own, tail, ambiguous = {}, {}, set()
        for p in phrases:
            r, k = role[id(p)]
            if r == 'ambiguous':
                ambiguous.update(k)
            elif r in ('suffix', 'label_next'):
                own.setdefault(k, []).append(p)
            elif r == 'tail':
                tail.setdefault(k, []).append(p)
        for pos, kind, obj in events:
            if kind == 'P':
                if role[id(obj)][0] == 'prefix':
                    current = obj
                continue
            j = obj
            res[j]['method_conflict'] = False
            if j in ambiguous:
                res[j]['method'], res[j]['method_ev'], res[j]['method_conflict'] = 'Unk', None, True
                continue
            if j in own:
                ps = own[j]
                m = combine_methods(set().union(*[p['kinds'] for p in ps]))
                res[j]['method'], res[j]['method_ev'] = (m or 'Unk'), ((ps[0]['s'], ps[-1]['e']) if m else None)
                res[j]['method_conflict'] = m is None
                continue
            cands = ([current] if current is not None else []) + tail.get(j, [])
            ms = {combine_methods(c['kinds']) for c in cands}
            if not cands:
                res[j]['method'], res[j]['method_ev'] = 'Unk', None
            elif len(ms) == 1 and None not in ms:
                c = cands[-1]
                res[j]['method'], res[j]['method_ev'] = ms.pop(), (c['s'], c['e'])
            else:
                res[j]['method'], res[j]['method_ev'], res[j]['method_conflict'] = 'Unk', None, True
    return res


# ── notation ────────────────────────────────────────────────────────────────
QUALITATIVE_RE = re.compile(r"^(?:CF|HM|PL|NPL|NLP|PL\s*\+|(?:no\s+)?(?:perception\s+of\s+light|light\s+perception)(?:\s+only)?"
                            r"|hand\s+movements?(?:\s+only)?|count(?:ing)?\s+fingers)", re.I)


def shape_notation(val):
    """Notation implied by the value's own shape, independent of any model:
    None when the shape alone can't decide (positive decimals)."""
    v = val.strip()
    if re.match(r"^20/\d", v):
        return 'snellen_ft'
    if re.match(r"^\d{1,2}(?:\.\d+)?/\d", v):
        return 'snellen_m'
    if QUALITATIVE_RE.match(v):
        return 'qualitative'
    if re.fullmatch(r"\d{1,3}(?:\s*letters)?", v, re.I):
        return 'etdrs'
    return None


def notation_for(text, s, e, prior_unit, scope_lo, scope_hi):
    """Keep determinate notations; for decimal/logmar/thomson require a
    chart named in scope (the value's segment, or table header lines)."""
    val = text[s:e].strip()
    if prior_unit is None:
        prior_unit = shape_notation(val)
    if prior_unit in ('snellen_m', 'snellen_ft', 'qualitative', 'etdrs'):
        return prior_unit, None
    if val.startswith('-'):
        return 'logmar', None
    best = None
    for m in CHART_RE.finditer(text, scope_lo, scope_hi):
        kind = next(k for k in ('logmar', 'thomson', 'decimal', 'etdrs') if m.group(k))
        if kind == 'etdrs':
            continue
        d = abs(m.start() - s)
        if best is None or d < best[0]:
            best = (d, kind, (m.start(), m.end()))
    if best:
        return best[1], best[2]
    return 'Unk', None


# ── entry point ─────────────────────────────────────────────────────────────
def label_document(text, spans, prior_units=None):
    """spans: list of (s, e). prior_units: optional list of the existing
    va_units values (used only to keep determinate notations).
    Returns list of dicts: lat, lat_ev, method, method_ev, notation,
    notation_ev, source ('table' | 'narrative')."""
    prior_units = prior_units or [None] * len(spans)
    cells = parse_tables(text)
    out = [None] * len(spans)
    narrative_idx = []
    for i, (s, e) in enumerate(spans):
        cell = next((c for c in cells if c['s'] <= s and e <= c['e']), None)
        if cell:
            out[i] = {'lat': cell['lat'], 'lat_ev': cell['lat_ev'], 'method': cell['method'],
                      'method_ev': cell['method_ev'], 'source': 'table', '_scope': (cell['header_lo'], cell['e'])}
        else:
            narrative_idx.append(i)
    if narrative_idx:
        nres = narrative_attributes(text, [spans[i] for i in narrative_idx])
        for i, r in zip(narrative_idx, nres):
            lo, hi = segment_bounds(text, spans[i][0])
            out[i] = {**r, 'source': 'narrative', '_scope': (lo, hi)}
    for i, (s, e) in enumerate(spans):
        lo, hi = out[i].pop('_scope')
        if out[i]['source'] == 'narrative':
            # a heading line just above can name the chart, e.g. "Visual Acuity (LogMAR):"
            lo = text.rfind('\n', 0, max(0, lo - 1)) + 1
        out[i]['notation'], out[i]['notation_ev'] = notation_for(text, s, e, prior_units[i], lo, max(hi, e))
    return out
