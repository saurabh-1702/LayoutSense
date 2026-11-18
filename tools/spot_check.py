"""Spot-check the dataset (project spec section 6.6, step 8).

Picks N random pages for every layout x tier cell and draws the answer-key boxes on them:
  - <out>/<layout>/<tier>/RXXXX_pN_words.jpg    every word coloured by its label (grey = unlabelled)
  - <out>/<layout>/<tier>/RXXXX_pN_regions.jpg  layout regions (header, sidebar, sections, tables ...)
  - <out>/sheet_<layout>_<tier>.jpg             contact sheet of all word overlays in that cell
  - <out>/legend.jpg                            colour legend for the word labels

    python tools\\spot_check.py --data dataset --out spot_check --per-cell 12
"""
import argparse
import csv
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TIERS = ["clean", "mild", "heavy"]


def colour(name):
    if name == "O":
        return (170, 170, 170)
    h = zlib.crc32(name.encode()) % 180
    hsv = np.uint8([[[h, 210, 235]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def overlay(img, ann, regions=False):
    over = img.copy()
    if regions:
        for r in ann["regions"]:
            c = colour(r["cls"])
            pts = np.int32(r["poly"]).reshape(-1, 1, 2)
            cv2.polylines(over, [pts], True, c, 7)
            x, y = pts[0, 0]
            cv2.putText(over, r["cls"], (int(x) + 10, int(y) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 4)
        return cv2.addWeighted(over, 0.85, img, 0.15, 0)
    for w in ann["words"]:
        lab = w["label"].split("-", 1)[-1]
        cv2.fillPoly(over, [np.int32(w["poly"]).reshape(-1, 1, 2)], colour(lab))
    return cv2.addWeighted(over, 0.4, img, 0.6, 0)


def small(img, h=900):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def sheet(images, cols=4, h=700):
    tiles = [small(i, h) for i in images]
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 6, 6, 6, 6 + w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def legend(labels):
    labels = sorted(labels)
    img = np.full((40 * len(labels) + 20, 420, 3), 255, np.uint8)
    for i, lab in enumerate(labels):
        y = 20 + 40 * i
        cv2.rectangle(img, (15, y), (55, y + 28), colour(lab), -1)
        cv2.putText(img, lab, (70, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="spot_check")
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["layout"], r["tier"])].append(r)
    rng = random.Random(a.seed)
    seen_labels, problems = set(), []
    for (layout, tier) in sorted(cells, key=lambda k: (k[0], TIERS.index(k[1]))):
        pick = rng.sample(cells[(layout, tier)], min(a.per_cell, len(cells[(layout, tier)])))
        d = out / layout / tier
        d.mkdir(parents=True, exist_ok=True)
        thumbs = []
        for r in pick:
            ann = json.loads((data / r["annotation"]).read_text(encoding="utf-8"))
            img = cv2.imread(str(data / r["image"]))
            if img is None:
                problems.append(f"cannot read {r['image']}")
                continue
            if (img.shape[1], img.shape[0]) != (ann["width"], ann["height"]):
                problems.append(f"size mismatch {r['image']}")
            # every word box must lie inside the image
            outside = sum(1 for w in ann["words"] if w["bbox"][0] < -2 or w["bbox"][1] < -2 or w["bbox"][2] > ann["width"] + 2 or w["bbox"][3] > ann["height"] + 2)
            if outside:
                problems.append(f"{outside} word boxes outside {r['image']}")
            seen_labels.update(w["label"].split("-", 1)[-1] for w in ann["words"])
            stem = Path(r["image"]).stem
            wimg = overlay(img, ann)
            cv2.imwrite(str(d / f"{stem}_words.jpg"), small(wimg, 1800), [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(str(d / f"{stem}_regions.jpg"), small(overlay(img, ann, regions=True), 1800), [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumbs.append(wimg)
        if thumbs:
            cv2.imwrite(str(out / f"sheet_{layout}_{tier}.jpg"), sheet(thumbs), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{layout:14s} {tier:6s} {len(pick):3d} pages checked")
    cv2.imwrite(str(out / "legend.jpg"), legend(seen_labels))
    print(f"\nOverlays written to {out.resolve()}")
    print("Automatic checks:", "all passed" if not problems else f"{len(problems)} problem(s)")
    for p in problems[:20]:
        print("  ", p)


if __name__ == "__main__":
    main()
