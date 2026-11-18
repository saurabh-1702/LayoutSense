"""Stage 1 + OCR: preprocess every image and run ONE OCR engine on it, saving words with boxes.

Both the NLP-only and the Layout-Aware branch read these files, so OCR quality is identical
for both (the control required by spec section 5).

    python pipeline\\run_ocr.py --data dataset --out ocr --workers 4
    python pipeline\\run_ocr.py --data dataset --out ocr --split test --tiers heavy --limit 20   (quick try)

Writes for every image:
  ocr/tesseract/<tier>/RXXXX_pN.json   words: text, conf, bbox (preprocessed image), bbox_orig/poly_orig
                                       (original image = same frame as dataset/annotations), line ids,
                                       text blocks from morphology+contours, preprocessing steps, matrix M
  ocr/tesseract/prep/<tier>/RXXXX_pN.jpg   the preprocessed (rectified, cleaned) page given to OCR
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per worker process

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.preprocess import preprocess, text_blocks  # noqa: E402

WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
USER_DEFAULT = r"D:\TesseractOCR\tesseract.exe"
TESS_CONFIG = "--oem 1 --psm 3"


def find_tesseract(user_path=None):
    for p in [user_path, shutil.which("tesseract"), USER_DEFAULT, WIN_DEFAULT, r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if p and Path(p).exists():
            return str(p)
    return None


def map_back(Minv, box):
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    poly = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return poly, [min(xs), min(ys), max(xs), max(ys)]


def ocr_one(args):
    row, data, out, tess, lang, overwrite = args
    tier = row["tier"]
    stem = Path(row["image"]).stem
    jpath = out / "tesseract" / tier / f"{stem}.json"
    if jpath.exists() and not overwrite:
        return stem, tier, "skipped"
    pytesseract.pytesseract.tesseract_cmd = tess
    t0 = time.time()
    img = cv2.imread(str(data / row["image"]))
    if img is None:
        return stem, tier, f"cannot read {row['image']}"
    pp = preprocess(img, tier)
    gray, M = pp["gray"], pp["M"]
    Minv = np.linalg.inv(M)
    d = pytesseract.image_to_data(gray, lang=lang, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
    words = []
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        conf = float(d["conf"][i])
        if not txt or conf < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        bbox = [x, y, x + w, y + h]
        poly, bb_orig = map_back(Minv, bbox)
        words.append(dict(id=len(words), text=txt, conf=round(conf, 1), bbox=bbox, bbox_orig=bb_orig, poly_orig=poly,
                          block=d["block_num"][i], par=d["par_num"][i], line=d["line_num"][i], word=d["word_num"][i]))
    # line ids that are unique across the page (Tesseract's own reading order)
    keys = {}
    for w in words:
        w["line_id"] = keys.setdefault((w["block"], w["par"], w["line"]), len(keys))
    prep_rel = Path("tesseract") / "prep" / tier / f"{stem}.jpg"
    (out / prep_rel).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / prep_rel), gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    doc = dict(image=row["image"], resume_id=row["resume_id"], page=int(row["page"]), tier=tier, layout=row["layout"],
               engine="tesseract", engine_version=str(pytesseract.get_tesseract_version()), config=TESS_CONFIG, lang=lang,
               preprocess=pp["info"], M=[[round(float(v), 8) for v in r] for r in M], prep_image=str(prep_rel).replace("\\", "/"),
               prep_width=int(gray.shape[1]), prep_height=int(gray.shape[0]), orig_width=int(img.shape[1]), orig_height=int(img.shape[0]),
               text_blocks=text_blocks(pp["binary"]), words=words, seconds=round(time.time() - t0, 2))
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return stem, tier, f"{len(words)} words in {doc['seconds']}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="ocr")
    ap.add_argument("--tiers", default="clean,mild,heavy")
    ap.add_argument("--split", default="all", help="all / train / val / test")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images per tier (quick test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--tesseract", default=None, help=r"path to tesseract.exe if not on PATH")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tess = find_tesseract(a.tesseract)
    if not tess:
        sys.exit("Tesseract not found. Install it (see README) or pass --tesseract \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"")
    data, out = Path(a.data), Path(a.out)
    tiers = a.tiers.split(",")
    rows = [r for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8"))
            if r["tier"] in tiers and (a.split == "all" or r["split"] == a.split)]
    if a.limit:
        rows = [r for t in tiers for r in [x for x in rows if x["tier"] == t][: a.limit]]
    print(f"Tesseract: {tess}  |  {len(rows)} images  |  {a.workers} workers")
    jobs = [(r, data, out, tess, a.lang, a.overwrite) for r in rows]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (stem, tier, msg) in enumerate(pool.imap_unordered(ocr_one, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {tier:5s} {stem}: {msg}", flush=True)
    print(f"\nDone in {time.time() - t0:.0f}s. Output in {out.resolve()}")


if __name__ == "__main__":
    main()
