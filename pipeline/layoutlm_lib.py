"""Shared code for Model B (LayoutLMv3): label projection, word ordering, windowing,
inference and decoding of word labels back into the answer-key JSON schema.

Used both locally (layoutlm_prep.py) and on Kaggle (the training notebook imports this file).
Only depends on the standard library at import time; torch/transformers are imported lazily.
"""
import re
from collections import defaultdict

ENTITY_TYPES = ["NAME", "HEADLINE", "EMAIL", "PHONE", "LOCATION", "LINK",
                "EDU_DEGREE", "EDU_INSTITUTION", "EDU_DATE", "EDU_GRADE", "EDU_LOCATION",
                "EXP_TITLE", "EXP_COMPANY", "EXP_DATE", "EXP_LOCATION", "EXP_DESC",
                "SKILL", "SKILL_CATEGORY", "SKILL_LEVEL",
                "PROJ_NAME", "PROJ_DESC", "PROJ_TECH", "PROJ_DATE",
                "CERT_NAME", "CERT_ISSUER", "CERT_DATE", "AWARD_NAME", "AWARD_ISSUER", "AWARD_DATE",
                "LANGUAGE", "LANG_LEVEL", "SECTION_HEADING"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ------------------------------------------------------------ word order --
def raster_order(words, box_key="bbox"):
    """Same layout-blind raster order Model A uses: lines across the full page width, left->right."""
    if not words:
        return []
    hs = sorted(w[box_key][3] - w[box_key][1] for w in words)
    mh = hs[len(hs) // 2] or 10
    idx = sorted(range(len(words)), key=lambda i: (words[i][box_key][1] + words[i][box_key][3]) / 2)
    lines, cur, cy = [], [], None
    for i in idx:
        b = words[i][box_key]
        c = (b[1] + b[3]) / 2
        if cur and abs(c - cy) > 0.5 * mh:
            lines.append(cur)
            cur = []
        cur.append(i)
        cy = c if len(cur) == 1 else (cy * (len(cur) - 1) + c) / len(cur)
    if cur:
        lines.append(cur)
    return [i for L in lines for i in sorted(L, key=lambda i: words[i][box_key][0])]


def tesseract_order(words):
    return sorted(range(len(words)), key=lambda i: (words[i]["line_id"], words[i]["word"]))


# ------------------------------------------------------ label projection --
def project_labels(ocr_words, gt_words):
    """Give every OCR word the field of the answer-key word it covers (box overlap, original-image coords).
    Returns list of (entity_type or None, field_path or None) aligned with ocr_words."""
    out = []
    for o in ocr_words:
        ob = o["bbox_orig"]
        best, best_v = None, 0.0
        for g in gt_words:
            gb = g["bbox"]
            ix = max(0.0, min(ob[2], gb[2]) - max(ob[0], gb[0]))
            iy = max(0.0, min(ob[3], gb[3]) - max(ob[1], gb[1]))
            inter = ix * iy
            if inter <= 0:
                continue
            ga = max(1e-6, (gb[2] - gb[0]) * (gb[3] - gb[1]))
            oa = max(1e-6, (ob[2] - ob[0]) * (ob[3] - ob[1]))
            v = inter / min(ga, oa)  # overlap relative to the smaller box (robust to OCR merges/splits)
            if v > best_v:
                best, best_v = g, v
        if best is not None and best_v >= 0.4 and best["label"] != "O":
            out.append((best["label"].split("-", 1)[1], best.get("field")))
        else:
            out.append((None, None))
    return out


def bio(seq):
    """seq: list of (entity_type, field_path) in reading order -> BIO label strings.
    B- marks the FIRST word of a field instance; every later word of the same instance is I-,
    even when words from another column are interleaved in between (raster order), so a value
    that wraps onto a second line is a continuation, not a new value."""
    labels, seen = [], set()
    for etype, field in seq:
        if etype is None:
            labels.append("O")
            continue
        key = field or etype
        labels.append(("I-" if key in seen else "B-") + etype)
        seen.add(key)
    return labels


def norm_box(b, w, h):
    x0, y0, x1, y1 = b
    f = lambda v, m: int(max(0, min(1000, round(1000 * v / m))))
    x0, x1 = sorted((f(x0, w), f(x1, w)))
    y0, y1 = sorted((f(y0, h), f(y1, h)))
    return [x0, y0, max(x1, x0 + 1) if x1 < 1000 else x1, max(y1, y0 + 1) if y1 < 1000 else y1]


# -------------------------------------------------------------- windows --
def make_windows(tokenizer, words, max_tokens=500, overlap_words=48):
    """Split a page's words into overlapping windows that fit LayoutLMv3's 512-token limit."""
    lens = [max(1, len(tokenizer.tokenize(" " + w))) for w in words]
    wins, start = [], 0
    while start < len(words):
        tot, end = 0, start
        while end < len(words) and tot + lens[end] <= max_tokens:
            tot += lens[end]
            end += 1
        end = max(end, start + 1)
        wins.append((start, end))
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)
    return wins


def encode_window(tokenizer, image_processor, image, words, boxes, labels=None):
    """Tokenise one window (version-robust: tokenizer and image processor called separately)."""
    kw = dict(boxes=boxes, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    if labels is not None:
        kw["word_labels"] = [LABEL2ID[l] for l in labels]
    enc = tokenizer(words, **kw)
    pix = image_processor(image, return_tensors="pt")["pixel_values"]
    return enc, pix


def predict_page(model, tokenizer, image_processor, image, words, boxes, device, max_tokens=500):
    """Returns one label string per word (best window = the one where the word is most central)."""
    import torch
    if not words:
        return []
    best = [None] * len(words)  # (centrality, label_id)
    for (s, e) in make_windows(tokenizer, words, max_tokens):
        enc, pix = encode_window(tokenizer, image_processor, image, words[s:e], boxes[s:e])
        word_ids = enc.word_ids(0)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                           bbox=enc["bbox"].to(device), pixel_values=pix.to(device)).logits[0]
        pred = logits.argmax(-1).tolist()
        seen = set()
        for tok_i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            gi = s + wid
            cent = min(wid, (e - s) - 1 - wid)
            if best[gi] is None or cent > best[gi][0]:
                best[gi] = (cent, pred[tok_i])
    return [ID2LABEL[b[1]] if b else "O" for b in best]


# -------------------------------------------------------------- decoding --
def _clean(s):
    s = re.sub(r"\s+", " ", s).strip(" ,;|•·:-–—[]")
    if s.count("(") > s.count(")"):
        s = s.rstrip("(").strip() if s.endswith("(") else s + ")" if re.search(r"\([^)]*$", s) else s
    if s.startswith(")"):
        s = s[1:].strip()
    return s


def spans(words, labels):
    """Field instances: B- starts one, I- continues the current one of the same type, or the most recent
    one of that type if other text came in between (wrapped value in another column)."""
    out = []  # [type, [words]]
    last_of = {}
    cur = None
    for w, l in zip(words, labels):
        if l == "O":
            cur = None
            continue
        p, t = l.split("-", 1)
        if p == "B" or t not in last_of:
            out.append([t, [w]])
            last_of[t] = cur = len(out) - 1
        elif cur is not None and out[cur][0] == t:
            out[cur][1].append(w)
        else:
            out[last_of[t]][1].append(w)
            cur = last_of[t]
    res = []
    for t, ws in out:
        txt = _clean(" ".join(ws))
        if txt:
            res.append((t, txt))
    return res


def _entries(sp, mapping, new_on, merge_adjacent=()):
    """Group spans into entries (one job, one degree ...).
    mapping: entity -> output key; new_on: keys that start a new entry when already filled."""
    entries, cur, last_key = [], {}, None
    for t, text in sp:
        if t not in mapping:
            continue
        key = mapping[t]
        if key == "bullets":
            cur.setdefault("bullets", []).append(text)
            last_key = key
            continue
        if key in cur:
            if key in merge_adjacent and last_key == key:
                cur[key] = f"{cur[key]} – {text}"  # "Jul 2021" + "Present" (separate From/To table cells)
                continue
            entries.append(cur)
            cur = {}
        cur[key] = text
        last_key = key
    if cur:
        entries.append(cur)
    anchors = set(new_on)
    return [e for e in entries if anchors & set(e)]


def decode(words, labels):
    """Word labels (whole resume, reading order) -> answer-key-schema JSON."""
    sp = spans(words, labels)
    first = lambda t: next((x for tt, x in sp if tt == t), None)
    res = {}
    if first("NAME"):
        res["name"] = first("NAME")
    if first("HEADLINE"):
        res["headline"] = first("HEADLINE")
    c = {}
    if first("EMAIL"):
        c["email"] = first("EMAIL").replace(" ", "")
    if first("PHONE"):
        c["phone"] = first("PHONE")
    if first("LOCATION"):
        c["location"] = first("LOCATION")
    if c:
        res["contact"] = c
    links = [x.replace(" ", "") for t, x in sp if t == "LINK"]
    if links:
        res["links"] = [{"url": u} for u in links]
    edu = _entries(sp, {"EDU_DEGREE": "degree", "EDU_INSTITUTION": "institution", "EDU_DATE": "dates", "EDU_GRADE": "grade",
                        "EDU_LOCATION": "location"}, new_on=("degree", "institution"), merge_adjacent=("dates",))
    if edu:
        res["education"] = edu
    exp = _entries(sp, {"EXP_TITLE": "title", "EXP_COMPANY": "company", "EXP_DATE": "dates", "EXP_LOCATION": "location",
                        "EXP_DESC": "bullets"}, new_on=("title", "company"), merge_adjacent=("dates",))
    if exp:
        res["experience"] = exp
    skills = []
    for t, x in sp:
        if t == "SKILL":
            for part in re.split(r"\s*[,|•·;]\s*", x):
                part = _clean(part)
                if part and part.lower() not in (s.lower() for s in skills):
                    skills.append(part)
    if skills:
        res["skills"] = skills
    for key, mp, new in (("projects", {"PROJ_NAME": "name", "PROJ_DESC": "description", "PROJ_TECH": "tech", "PROJ_DATE": "dates"}, ("name",)),
                         ("certifications", {"CERT_NAME": "name", "CERT_ISSUER": "issuer", "CERT_DATE": "year"}, ("name",)),
                         ("awards", {"AWARD_NAME": "name", "AWARD_ISSUER": "issuer", "AWARD_DATE": "year"}, ("name",)),
                         ("languages", {"LANGUAGE": "language", "LANG_LEVEL": "proficiency"}, ("language",))):
        ents = _entries(sp, mp, new_on=new)
        ents = [e for e in ents if any(k in e for k in new)]
        if ents:
            res[key] = ents
    return res
