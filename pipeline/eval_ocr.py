"""Measure OCR quality on its own (spec section 15: OCR errors are reported separately,
not blamed on either model).

Each OCR word (mapped back to the original image) is matched to an answer-key word by box
overlap; reading order does not matter. Per layout x tier it reports:
  detection recall   share of answer-key words that OCR found at all
  word accuracy      share of answer-key words found AND read exactly (case-insensitive)
  CER                character error rate over all answer-key words (missed words count fully)
  field ceiling      share of field values (e.g. each company name) whose OCR text is exactly right,
                     ignoring spaces/case = the best ANY parser using this OCR could do on that field
Punctuation-only glyphs (bullets, separators) are excluded from the word-level numbers.

    python pipeline\\eval_ocr.py --data dataset --ocr ocr
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

GROUPS = {"NAME": "name", "EMAIL": "email", "PHONE": "phone", "LOCATION": "location", "LINK": "links",
          "EDU_DEGREE": "education", "EDU_INSTITUTION": "education", "EDU_DATE": "education", "EDU_GRADE": "education",
          "EXP_TITLE": "experience", "EXP_COMPANY": "experience", "EXP_DATE": "experience", "EXP_LOCATION": "experience",
          "EXP_DESC": "exp_bullets", "SKILL": "skills", "CERT_NAME": "certifications", "CERT_ISSUER": "certifications",
          "LANGUAGE": "languages", "PROJ_NAME": "projects", "PROJ_DESC": "projects"}
FIELD_COLS = ["name", "email", "phone", "location", "education", "experience", "exp_bullets", "skills", "certifications", "languages"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def norm(t):
    return t.strip().lower()


def squash(t):
    return "".join(t.split()).lower()


def field_ceiling(gt_words, ocr_words):
    """Per field instance (e.g. experience.0.company): is the OCR text over that field exactly
    the printed value, ignoring spaces and case? Robust to OCR merging/splitting words."""
    by_field = defaultdict(list)
    for g in gt_words:
        if g.get("field"):
            by_field[g["field"]].append(g)
    res = defaultdict(lambda: [0, 0])
    for path, gws in by_field.items():
        grp = GROUPS.get(gws[0]["label"].split("-", 1)[-1])
        if not grp:
            continue
        boxes = [g["bbox"] for g in gws]
        got = []
        for o in ocr_words:
            b = o["bbox_orig"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(c.isalnum() for c in o["text"]) and any(x0 - 3 <= cx <= x1 + 3 and y0 - 3 <= cy <= y1 + 3 for x0, y0, x1, y1 in boxes):
                got.append(o["text"])
        truth = squash("".join(g["text"] for g in gws if any(c.isalnum() for c in g["text"])))
        res[grp][0] += int(squash("".join(got)) == truth)
        res[grp][1] += 1
    return res


def eval_page(gt_words, ocr_words):
    fc = field_ceiling(gt_words, ocr_words)
    # punctuation-only glyphs (bullets, separators) are not scored at word level
    gt_words = [g for g in gt_words if any(c.isalnum() for c in g["text"])]
    # candidate pairs by overlap, greedy one-to-one
    pairs = []
    for i, g in enumerate(gt_words):
        for j, o in enumerate(ocr_words):
            if o["bbox_orig"][0] > g["bbox"][2] or o["bbox_orig"][2] < g["bbox"][0] or o["bbox_orig"][1] > g["bbox"][3] or o["bbox_orig"][3] < g["bbox"][1]:
                continue
            v = iou(g["bbox"], o["bbox_orig"])
            if v >= 0.3:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    mg, mo, match = set(), set(), {}
    for v, i, j in pairs:
        if i in mg or j in mo:
            continue
        mg.add(i), mo.add(j)
        match[i] = j
    chars = edits = correct = 0
    for i, g in enumerate(gt_words):
        gt = norm(g["text"])
        chars += len(gt)
        ok = False
        if i in match:
            ot = norm(ocr_words[match[i]]["text"])
            edits += min(len(gt), Levenshtein.distance(gt, ot))
            ok = gt == ot
        else:
            edits += len(gt)
        correct += ok
    per_field = fc
    n = len(gt_words)
    return dict(gt_words=n, ocr_words=len(ocr_words), detected=len(match), correct=correct, chars=chars, edits=edits,
                per_field={k: tuple(v) for k, v in per_field.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ocr", default="ocr")
    ap.add_argument("--engine", default="tesseract")
    a = ap.parse_args()
    data, ocr = Path(a.data), Path(a.ocr) / a.engine
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    per_image, cell = [], defaultdict(lambda: defaultdict(float))
    for r in rows:
        jp = ocr / r["tier"] / f"{Path(r['image']).stem}.json"
        if not jp.exists():
            continue
        o = json.loads(jp.read_text(encoding="utf-8"))
        g = json.loads((data / r["annotation"]).read_text(encoding="utf-8"))
        m = eval_page(g["words"], o["words"])
        rec = dict(image=r["image"], layout=r["layout"], tier=r["tier"], split=r["split"], gt_words=m["gt_words"], ocr_words=m["ocr_words"],
                   detection_recall=round(m["detected"] / max(1, m["gt_words"]), 4), word_accuracy=round(m["correct"] / max(1, m["gt_words"]), 4),
                   cer=round(m["edits"] / max(1, m["chars"]), 4), preprocess=" > ".join(o["preprocess"]["steps"]))
        for f in FIELD_COLS:
            c, t = m["per_field"].get(f, (0, 0))
            rec[f"ceiling_{f}"] = round(c / t, 4) if t else ""
        per_image.append(rec)
        k = cell[(r["layout"], r["tier"])]
        for key in ("gt_words", "detected", "correct", "chars", "edits"):
            k[key] += m[key]
        for f, (c, t) in m["per_field"].items():
            k[f"c_{f}"] += c
            k[f"t_{f}"] += t
        k["pages"] += 1
    if not per_image:
        raise SystemExit(f"No OCR files found in {ocr}. Run pipeline/run_ocr.py first.")
    out = Path(a.ocr)
    with open(out / "ocr_eval_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    summ = []
    tiers = ["clean", "mild", "heavy"]
    print(f"{'layout':14s} {'tier':6s} {'pages':>5s} {'detect':>7s} {'wordacc':>7s} {'CER':>6s} | field ceilings: " + " ".join(f"{c[:7]:>7s}" for c in FIELD_COLS))
    for (layout, tier) in sorted(cell, key=lambda x: (x[0], tiers.index(x[1]))):
        k = cell[(layout, tier)]
        s = dict(layout=layout, tier=tier, pages=int(k["pages"]), detection_recall=k["detected"] / k["gt_words"],
                 word_accuracy=k["correct"] / k["gt_words"], cer=k["edits"] / k["chars"])
        for f in FIELD_COLS:
            s[f"ceiling_{f}"] = (k[f"c_{f}"] / k[f"t_{f}"]) if k[f"t_{f}"] else None
        summ.append(s)
        print(f"{layout:14s} {tier:6s} {s['pages']:5d} {s['detection_recall']:7.3f} {s['word_accuracy']:7.3f} {s['cer']:6.3f} | "
              + " ".join(f"{s[f'ceiling_{f}']:7.3f}" if s[f"ceiling_{f}"] is not None else f"{'-':>7s}" for f in FIELD_COLS))
    with open(out / "ocr_eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in summ])
    print(f"\nWrote {out / 'ocr_eval_per_image.csv'} and {out / 'ocr_eval_summary.csv'}")


if __name__ == "__main__":
    main()
