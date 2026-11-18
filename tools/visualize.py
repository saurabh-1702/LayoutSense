"""Draw the answer-key boxes on top of an image to sanity-check alignment.

    python tools/visualize.py dataset annotations/heavy/R0007_p1.json  out.png
    python tools/visualize.py dataset annotations/heavy/R0007_p1.json  out.png --regions
"""
import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np


def colour(name):
    h = zlib.crc32(name.encode()) % 180
    hsv = np.uint8([[[h, 200, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("annotation")
    ap.add_argument("out")
    ap.add_argument("--regions", action="store_true", help="draw layout regions instead of words")
    a = ap.parse_args()
    root = Path(a.root)
    ann = json.loads((root / a.annotation).read_text(encoding="utf-8"))
    img = cv2.imread(str(root / ann["image"]))
    over = img.copy()
    items = ann["regions"] if a.regions else ann["words"]
    for it in items:
        key = it["cls"] if a.regions else it["label"].split("-")[-1]
        if key == "O" and not a.regions:
            c = (160, 160, 160)
        else:
            c = colour(key)
        pts = np.int32(it["poly"]).reshape(-1, 1, 2)
        if a.regions:
            cv2.polylines(over, [pts], True, c, 6)
            cv2.putText(over, key, tuple(pts[0, 0] + [8, 40]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, c, 3)
        else:
            cv2.fillPoly(over, [pts], c)
    out = cv2.addWeighted(over, 0.35 if not a.regions else 0.8, img, 0.65 if not a.regions else 0.2, 0)
    cv2.imwrite(a.out, out)


if __name__ == "__main__":
    main()
