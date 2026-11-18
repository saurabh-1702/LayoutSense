"""Turn raw browser tokens into OCR-like words, lines, labels, gold text and the
field-level answer key."""
import re

from .render import SCALE

# coarse token label for each field path (used for LayoutLM-style BIO tags)
LABEL_RULES = [
    (r"^name$", "NAME"), (r"^headline$", "HEADLINE"), (r"^contact\.email$", "EMAIL"), (r"^contact\.phone$", "PHONE"),
    (r"^contact\.location$", "LOCATION"), (r"^links\.\d+\.url$", "LINK"),
    (r"^education\.\d+\.degree$", "EDU_DEGREE"), (r"^education\.\d+\.institution$", "EDU_INSTITUTION"),
    (r"^education\.\d+\.(dates|year)$", "EDU_DATE"), (r"^education\.\d+\.grade$", "EDU_GRADE"), (r"^education\.\d+\.location$", "EDU_LOCATION"),
    (r"^experience\.\d+\.title$", "EXP_TITLE"), (r"^experience\.\d+\.company$", "EXP_COMPANY"),
    (r"^experience\.\d+\.(dates|start_text|end_text)$", "EXP_DATE"), (r"^experience\.\d+\.location$", "EXP_LOCATION"),
    (r"^experience\.\d+\.bullets\.\d+$", "EXP_DESC"), (r"^skills\.\d+$", "SKILL"), (r"^skill_group\.\d+$", "SKILL_CATEGORY"),
    (r"^skill_level\.\d+$", "SKILL_LEVEL"),
    (r"^projects\.\d+\.name$", "PROJ_NAME"), (r"^projects\.\d+\.description$", "PROJ_DESC"), (r"^projects\.\d+\.tech$", "PROJ_TECH"),
    (r"^projects\.\d+\.dates$", "PROJ_DATE"),
    (r"^certifications\.\d+\.name$", "CERT_NAME"), (r"^certifications\.\d+\.issuer$", "CERT_ISSUER"), (r"^certifications\.\d+\.year$", "CERT_DATE"),
    (r"^awards\.\d+\.name$", "AWARD_NAME"), (r"^awards\.\d+\.issuer$", "AWARD_ISSUER"), (r"^awards\.\d+\.year$", "AWARD_DATE"),
    (r"^languages\.\d+\.language$", "LANGUAGE"), (r"^languages\.\d+\.proficiency$", "LANG_LEVEL"),
    (r"^heading\.", "SECTION_HEADING"),
]
_LABEL_RX = [(re.compile(a), b) for a, b in LABEL_RULES]


def label_of(field):
    if not field:
        return "O"
    for rx, lab in _LABEL_RX:
        if rx.search(field):
            return lab
    return "O"


def _alnum(s):
    return sum(c.isalnum() for c in s)


def build_words(tokens):
    """Merge DOM tokens that touch visually (e.g. "Python" + ",") into one OCR-style word."""
    words = []
    for t in tokens:
        h = t["y1"] - t["y0"]
        if words:
            w = words[-1]
            same_line = abs((t["y0"] + t["y1"]) / 2 - (w["y0"] + w["y1"]) / 2) < 0.35 * h
            gap = t["x0"] - w["x1"]
            if same_line and -1.0 <= gap < 0.12 * t["fs"]:
                w["text"] += t["t"]
                w["x1"], w["y0"], w["y1"] = max(w["x1"], t["x1"]), min(w["y0"], t["y0"]), max(w["y1"], t["y1"])
                w["_parts"].append(t)
                continue
        words.append(dict(text=t["t"], x0=t["x0"], y0=t["y0"], x1=t["x1"], y1=t["y1"], fs=t["fs"], blk=t["blk"], _parts=[t]))
    for w in words:
        best = max(w["_parts"], key=lambda p: (_alnum(p["t"]), p["field"] is not None))
        w["field"] = best["field"]
        w["label"] = label_of(w["field"])
        w["bold"] = best["bold"]
        del w["_parts"]
    return words


def build_lines(words):
    lines = []
    for i, w in enumerate(words):
        h = w["y1"] - w["y0"]
        if lines:
            L = lines[-1]
            pw = words[L["word_ids"][-1]]
            same_y = abs((w["y0"] + w["y1"]) / 2 - (pw["y0"] + pw["y1"]) / 2) < 0.5 * h
            forward = w["x0"] >= pw["x1"] - 1
            near = (w["x0"] - pw["x1"]) < 3.0 * w["fs"] or (w["blk"] is not None and w["blk"] == pw["blk"])
            if same_y and forward and near:
                L["word_ids"].append(i)
                continue
        lines.append(dict(word_ids=[i]))
    for L in lines:
        ws = [words[i] for i in L["word_ids"]]
        L["text"] = " ".join(w["text"] for w in ws)
        L["x0"], L["y0"] = min(w["x0"] for w in ws), min(w["y0"] for w in ws)
        L["x1"], L["y1"] = max(w["x1"] for w in ws), max(w["y1"] for w in ws)
    return lines


def bio(words):
    prev_field = None
    for w in words:
        if w["label"] == "O":
            w["bio"] = "O"
            prev_field = None
            continue
        w["bio"] = ("I-" if w["field"] == prev_field else "B-") + w["label"]
        prev_field = w["field"]


def scale_box(d, k=SCALE):
    return [round(d["x0"] * k, 1), round(d["y0"] * k, 1), round(d["x1"] * k, 1), round(d["y1"] * k, 1)]


def page_annotation(pdata):
    words = build_words(pdata["tokens"])
    bio(words)
    lines = build_lines(words)
    out_words = [dict(id=i, text=w["text"], bbox=scale_box(w), label=w["bio"], field=w["field"], line=None) for i, w in enumerate(words)]
    out_lines = []
    for li, L in enumerate(lines):
        for wi in L["word_ids"]:
            out_words[wi]["line"] = li
        out_lines.append(dict(id=li, text=L["text"], bbox=scale_box(L), word_ids=L["word_ids"]))
    regions = []
    for r in pdata["regions"]:
        cls = r["cls"]
        regions.append(dict(cls=cls, bbox=scale_box(r)))
    text = "\n".join(L["text"] for L in out_lines)
    return dict(words=out_words, lines=out_lines, regions=regions), text


# ---------------------------------------------------------------------------
def _set(d, parts, value):
    cur = d
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        nxt = parts[i + 1] if not last else None
        key = int(p) if p.isdigit() else p
        if isinstance(cur, list):
            while len(cur) <= key:
                cur.append(None)
            if last:
                if cur[key] is None:
                    cur[key] = value
                return
            if cur[key] is None:
                cur[key] = [] if nxt.isdigit() else {}
            cur = cur[key]
        else:
            if last:
                cur.setdefault(key, value)
                return
            if key not in cur:
                cur[key] = [] if nxt.isdigit() else {}
            cur = cur[key]


KEY_ORDER = ["resume_id", "name", "headline", "contact", "links", "education", "experience", "skills", "projects", "certifications",
             "awards", "languages", "title", "company", "degree", "institution", "name", "language", "proficiency", "issuer", "location", "dates",
             "start_text", "end_text", "year", "grade", "description", "tech", "url", "bullets", "start", "end", "level", "type", "phone_e164", "city", "country"]


def _ordered(d):
    if isinstance(d, dict):
        ks = sorted(d, key=lambda k: KEY_ORDER.index(k) if k in KEY_ORDER else 99)
        return {k: _ordered(d[k]) for k in ks}
    if isinstance(d, list):
        return [_ordered(x) for x in d]
    return d


def answer_key(person, pages):
    return _ordered(_answer_key(person, pages))


def _answer_key(person, pages):
    """Field-level answer key built ONLY from what is printed on the page."""
    seen = {}
    for pdata in pages:
        for f in pdata["fields"]:
            if f["path"].startswith(("heading.", "skill_group.", "skill_level.")):
                continue
            if f["path"] not in seen and f["text"]:
                seen[f["path"]] = f["text"]
    gt = {}
    for path in sorted(seen, key=lambda p: [int(x) if x.isdigit() else x for x in p.split(".")]):
        _set(gt, path.split("."), seen[path])
    # compact lists (None holes cannot happen unless a template skips an index)
    for k in ("education", "experience", "projects", "certifications", "awards", "languages", "links", "skills"):
        if k in gt:
            gt[k] = [x for x in gt[k] if x is not None]
    # ordered output + normalised metadata
    out = {"resume_id": person["resume_id"]}
    for k in ("name", "headline"):
        if k in gt:
            out[k] = gt[k]
    if "contact" in gt:
        c = dict(gt["contact"])
        if "phone" in c:
            c["phone_e164"] = person["contact"]["phone_e164"]
        if "location" in c:
            c["city"], c["country"] = person["contact"]["city"], person["contact"]["country"]
        out["contact"] = c
    if "links" in gt:
        out["links"] = [dict(type=person["links"][i]["type"], **l) for i, l in enumerate(gt["links"])]
    if "education" in gt:
        eds = []
        for i, e in enumerate(gt["education"]):
            src = person["education"][i]
            e = dict(e)
            e["start"] = src["start"]
            e["end"] = src["end"]
            e["level"] = {"b": "bachelor", "m": "master", "school": "school"}.get(src["kind"], src["kind"])
            eds.append(e)
        out["education"] = eds
    if "experience" in gt:
        exs = []
        for i, e in enumerate(gt["experience"]):
            src = person["experience"][i]
            e = dict(e)
            if "bullets" in e:
                e["bullets"] = [b for b in e["bullets"] if b is not None]
            e["start"], e["end"] = src["start"], src["end"]
            exs.append(e)
        out["experience"] = exs
    for k in ("skills", "projects", "certifications", "awards", "languages"):
        if k in gt:
            out[k] = gt[k]
    return out


def check_against_content(person, pages):
    """Every printed field must equal the content value (case-insensitive). Returns mismatches."""
    bad = []
    for pdata in pages:
        for f in pdata["fields"]:
            path = f["path"]
            if path.startswith(("heading.", "skill_group.", "skill_level.")):
                continue
            parts = path.split(".")
            cur = person
            try:
                for p in parts:
                    cur = cur[int(p)] if p.isdigit() else cur[p]
            except (KeyError, IndexError, TypeError):
                bad.append((path, "missing_in_content"))
                continue
            if not isinstance(cur, str):
                continue
            if re.sub(r"\s+", " ", cur).strip().lower() != f["text"].lower():
                bad.append((path, cur, f["text"]))
    return bad
