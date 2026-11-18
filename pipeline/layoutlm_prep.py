"""Model B, step 1 (run locally): package the data for LayoutLMv3 training on Kaggle.

Each training example = one page: the SAME OCR words Model A reads (from run_ocr.py), in the SAME
raster order, plus their boxes (0-1000) and the preprocessed page image. Labels come from the answer
key: every OCR word takes the field of the answer-key word it overlaps.

    python pipeline\\layoutlm_prep.py --data dataset --ocr ocr --out layoutlm_data
    python pipeline\\layoutlm_prep.py ... --check     (also writes preds\\label_ceiling = decoding of the
                                                     projected labels; score it to see the best a perfect
                                                     LayoutLM could do with this OCR)

Then zip layoutlm_data and upload it to Kaggle as a private dataset.
"""
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.layoutlm_lib import LABELS, bio, decode, norm_box, project_labels, raster_order, tesseract_order  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ocr", default="ocr")
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--out", default="layoutlm_data")
    ap.add_argument("--order", default="raster", choices=["raster", "tesseract"])
    ap.add_argument("--image-size", type=int, default=1000, help="longest side of the saved page images (model resizes to 224 anyway)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-out", default="preds/label_ceiling")
    a = ap.parse_args()
    data, ocr, out = Path(a.data), Path(a.ocr) / a.engine, Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    n, missing, stats = 0, 0, defaultdict(int)
    per_resume = defaultdict(list)
    with open(out / "examples.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            stem = Path(r["image"]).stem
            jp = ocr / r["tier"] / f"{stem}.json"
            if not jp.exists():
                missing += 1
                continue
            o = json.loads(jp.read_text(encoding="utf-8"))
            g = json.loads((data / r["annotation"]).read_text(encoding="utf-8"))
            words = o["words"]
            order = raster_order(words) if a.order == "raster" else tesseract_order(words)
            proj = project_labels(words, g["words"])
            seq = [proj[i] for i in order]
            labels = bio(seq)
            texts = [words[i]["text"] for i in order]
            boxes = [norm_box(words[i]["bbox"], o["prep_width"], o["prep_height"]) for i in order]
            orig_boxes = [words[i]["bbox_orig"] for i in order]
            img_rel = f"images/{r['tier']}/{stem}.jpg"
            dst = out / img_rel
            if not dst.exists():
                img = cv2.imread(str(Path(a.ocr) / o["prep_image"]), cv2.IMREAD_GRAYSCALE)
                h, w = img.shape
                s = a.image_size / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            ex = dict(id=f"{stem}_{r['tier']}", resume_id=r["resume_id"], page=int(r["page"]), tier=r["tier"], split=r["split"],
                      layout=r["layout"], template=r["template"], image=img_rel, words=texts, boxes=boxes, labels=labels,
                      orig_boxes=orig_boxes)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            per_resume[(r["resume_id"], r["tier"])].append(ex)
            stats[r["split"]] += 1
            n += 1
    (out / "labels.json").write_text(json.dumps(LABELS, indent=0))
    shutil.copy(HERE / "layoutlm_lib.py", out / "layoutlm_lib.py")
    nb = HERE / "train_layoutlmv3_kaggle.ipynb"
    if nb.exists():
        shutil.copy(nb, out / nb.name)
    print(f"{n} pages written to {out.resolve()}  ({dict(stats)});  {missing} pages without OCR skipped")
    if a.check:
        co = Path(a.check_out)
        for (rid, tier), exs in per_resume.items():
            exs.sort(key=lambda e: e["page"])
            words = [w for e in exs for w in e["words"]]
            labels = [l for e in exs for l in e["labels"]]
            (co / tier).mkdir(parents=True, exist_ok=True)
            (co / tier / f"{rid}.json").write_text(json.dumps({"resume_id": rid, **decode(words, labels)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Label ceiling predictions written to {co.resolve()}  ->  python score.py --data {a.data} --pred {co} --split test")


if __name__ == "__main__":
    main()
