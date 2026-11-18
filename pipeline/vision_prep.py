"""Build the Kaggle upload for the two vision deliverables (YOLO region detector + CNN layout classifier).

    python pipeline\\vision_prep.py --data dataset --out vision_data

Writes vision_data\\ (and vision_data.zip, the file you upload to Kaggle):
  yolo\\images\\{train,val,test}\\<tier>_<RID>_p<N>.jpg   page images, long side 1280 px (labels are normalised, so unchanged)
  yolo\\labels\\{train,val,test}\\<tier>_<RID>_p<N>.txt   region boxes copied from dataset\\yolo\\labels
  classes.txt                                         YOLO class names
  index.csv                                           one row per page image: split, tier, layout, template, resume_id, page
The CNN classifier uses the same images; its label is the layout column of index.csv.
Splits are the dataset's own: test = the 4 held-out templates, so both models are tested on unseen designs.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2

LONG_SIDE = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="vision_data")
    ap.add_argument("--long-side", type=int, default=LONG_SIDE)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    for s in ("train", "val", "test"):
        (out / "yolo" / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "yolo" / "labels" / s).mkdir(parents=True, exist_ok=True)
    shutil.copy(data / "yolo" / "classes.txt", out / "classes.txt")
    names = (data / "yolo" / "classes.txt").read_text().split()
    index, n_boxes = [], 0
    for i, r in enumerate(rows, 1):
        src = data / r["image"]
        stem = Path(r["image"]).stem                      # R0001_p1
        name = f"{r['tier']}_{stem}"
        img = cv2.imread(str(src))
        if img is None:
            print("  could not read", src)
            continue
        h, w = img.shape[:2]
        f = a.long_side / max(h, w)
        if f < 1:
            img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "yolo" / "images" / r["split"] / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # labels rebuilt from the page annotation (the same boxes the parsers were scored on), not copied from
        # dataset/yolo/labels, so they always belong to exactly this image
        ann = json.loads((data / r["annotation"]).read_text(encoding="utf-8"))
        iw, ih = ann["width"], ann["height"]
        lines = []
        for g in ann["regions"]:
            bb = g["bbox"]
            x0, y0, x1, y1 = max(0, bb[0]), max(0, bb[1]), min(iw, bb[2]), min(ih, bb[3])
            if x1 - x0 > 2 and y1 - y0 > 2:
                lines.append(f"{names.index(g['cls'])} {(x0 + x1) / 2 / iw:.6f} {(y0 + y1) / 2 / ih:.6f} {(x1 - x0) / iw:.6f} {(y1 - y0) / ih:.6f}")
        txt = "\n".join(lines) + "\n"
        if (w, h) != (iw, ih):
            print(f"  WARNING size mismatch {src}: image {w}x{h}, annotation {iw}x{ih}")
        n_boxes += sum(1 for L in txt.splitlines() if L.strip())
        (out / "yolo" / "labels" / r["split"] / f"{name}.txt").write_text(txt, encoding="utf-8")
        index.append(dict(file=f"yolo/images/{r['split']}/{name}.jpg", split=r["split"], tier=r["tier"], layout=r["layout"],
                          template=r["template"], resume_id=r["resume_id"], page=r["page"]))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} pages")
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        wr.writeheader()
        wr.writerows(index)
    cnt = {}
    for r in index:
        cnt[r["split"]] = cnt.get(r["split"], 0) + 1
    print(f"{len(index)} page images ({cnt}), {n_boxes} region boxes")
    if not a.no_zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"Upload this file to Kaggle as a new dataset: {Path(z).resolve()}  ({Path(z).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
