"""Check that dataset\\yolo\\labels agree with the region boxes in dataset\\annotations (which the parsers used).

    python tools\\check_yolo_labels.py --data dataset

Prints, per tier, how many label files match their annotation. Also draws two pages per tier to
report_figures\\label_check_<tier>.jpg: GREEN = boxes from the annotation, RED = boxes from the YOLO label file.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


def ann_lines(ann, names):
    iw, ih = ann["width"], ann["height"]
    out = []
    for r in ann["regions"]:
        bb = r["bbox"]
        x0, y0, x1, y1 = max(0, bb[0]), max(0, bb[1]), min(iw, bb[2]), min(ih, bb[3])
        if x1 - x0 > 2 and y1 - y0 > 2:
            out.append((names.index(r["cls"]), (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih, (x1 - x0) / iw, (y1 - y0) / ih))
    return out


def file_lines(p):
    if not p.exists():
        return None
    return [(int(L.split()[0]), *map(float, L.split()[1:5])) for L in p.read_text().splitlines() if L.strip()]


def same(a, b, tol=0.01):
    if a is None or len(a) != len(b):
        return False
    return all(x[0] == y[0] and max(abs(u - v) for u, v in zip(x[1:], y[1:])) < tol for x, y in zip(sorted(a), sorted(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="report_figures")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = (data / "yolo" / "classes.txt").read_text().split()
    stats = defaultdict(lambda: [0, 0, 0])   # tier -> [pages, match, missing]
    drawn = defaultdict(int)
    bad_examples = defaultdict(list)
    for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8")):
        tier, stem = r["tier"], Path(r["image"]).stem
        ann = json.loads((data / r["annotation"]).read_text(encoding="utf-8"))
        want = ann_lines(ann, names)
        have = file_lines(data / "yolo" / "labels" / tier / f"{stem}.txt")
        s = stats[tier]
        s[0] += 1
        if have is None:
            s[2] += 1
        elif same(have, want):
            s[1] += 1
        elif len(bad_examples[tier]) < 3:
            bad_examples[tier].append(stem)
        if drawn[tier] < 2 and r["layout"] == "multi_column":
            img = cv2.imread(str(data / r["image"]))
            h, w = img.shape[:2]
            for lines, col, th in ((want, (0, 200, 0), 9), (have or [], (0, 0, 255), 4)):
                for c, x, y, bw, bh in lines:
                    cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, th)
            img = cv2.resize(img, (900, int(900 * h / w)))
            cv2.putText(img, f"{tier} {stem}  image {w}x{h}  annotation {ann['width']}x{ann['height']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imwrite(str(out / f"label_check_{tier}_{drawn[tier] + 1}.jpg"), img)
            drawn[tier] += 1
    print(f"{'tier':6} {'pages':>6} {'match':>6} {'missing':>8}  examples of mismatches")
    for t in ("clean", "mild", "heavy"):
        if t in stats:
            p, m, miss = stats[t]
            print(f"{t:6} {p:6d} {m:6d} {miss:8d}  {', '.join(bad_examples[t])}")
    print(f"\nPictures: {out.resolve()}\\label_check_*.jpg  (green = annotation, red = YOLO label file)")


if __name__ == "__main__":
    main()
