"""Generate the synthetic resume dataset.

    python generate.py --n 300 --out dataset --workers 4

Produces, for every resume:
  images/{clean,mild,heavy}/RXXXX_pN.png       300-DPI page images
  ground_truth/RXXXX.json                       field-level answer key (shared by all tiers)
  text/RXXXX.txt, text/RXXXX_pN.txt             gold text in reading order
  annotations/{tier}/RXXXX_pN.json              word / line / region boxes + labels (tier-specific coords)
  yolo/labels/{tier}/RXXXX_pN.txt               region boxes in YOLO format
  html/RXXXX.html                               source HTML (for debugging / re-rendering)
and a manifest.csv with one row per image.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from resumegen.annotate import answer_key, check_against_content, page_annotation  # noqa: E402
from resumegen.content import PersonGen  # noqa: E402
from resumegen.degrade import TIERS, degrade, transform_box  # noqa: E402
from resumegen.render import SCALE, Renderer  # noqa: E402
from resumegen.style import LAYOUTS, TEMPLATE_LAYOUT, make_style  # noqa: E402

# Templates never seen in training: the test split uses only these (prevents layout memorisation).
TEST_TEMPLATES = {"s3_banded", "m3_two_equal", "t3_form", "i3_timeline_gauges"}

DEGRADED_PNG = False  # set by --degraded-png

YOLO_CLASSES = ["header", "sidebar", "column", "table", "section_experience", "section_education", "section_skills",
                "section_projects", "section_certifications", "section_awards", "section_languages", "section_contact",
                "section_links", "section_other"]


def region_class(cls):
    if cls in ("header", "sidebar", "column"):
        return cls
    if cls.startswith("tbl_"):
        return "table"
    if cls.startswith("sec_"):
        name = "section_" + cls[4:]
        return name if name in YOLO_CLASSES else "section_other"
    return "section_other"


LEVELS = [("fresher", 18), ("junior", 25), ("mid", 25), ("senior", 20), ("exec", 12)]
TWO_PAGE_P = {"fresher": 0.0, "junior": 0.1, "mid": 0.35, "senior": 0.65, "exec": 0.9}


def plan(n, seed):
    """Balanced plan: equal resumes per layout, templates rotated within a layout,
    seniority drawn per resume, and a target page count that follows seniority."""
    rng = random.Random(seed)
    layouts = list(LAYOUTS)
    jobs = []
    for i in range(n):
        layout = layouts[i % 4]
        tpl = LAYOUTS[layout][(i // 4) % 3]
        tot = sum(w for _, w in LEVELS)
        x, level = rng.uniform(0, tot), LEVELS[-1][0]
        for lv, w in LEVELS:
            x -= w
            if x <= 0:
                level = lv
                break
        pages = 2 if rng.random() < TWO_PAGE_P[level] else 1
        jobs.append(dict(idx=i, rid=f"R{i + 1:04d}", layout=layout, template=tpl, level=level, target_pages=pages, seed=seed * 100003 + i))
    rng.shuffle(jobs)
    return jobs


def split_of(rid, template):
    if template in TEST_TEMPLATES:
        return "test"
    return "val" if zlib.crc32(rid.encode()) % 100 < 15 else "train"


_R = None


def _renderer():
    global _R
    if _R is None:
        _R = Renderer()
    return _R


def page2_fill(pdata):
    if not pdata["tokens"]:
        return 0
    return max(t["y1"] for t in pdata["tokens"]) / pdata["height"]


def make_one(job, out):
    R = _renderer()
    out = Path(out)
    rid, tpl = job["rid"], job["template"]
    attempt_seed = job["seed"]
    target = job.get("target_pages", 1)
    res = person = style = html = None
    chosen = fallback = None
    for attempt in range(3):
        for richness in (range(0, 5) if target == 2 else [0]):
            person = PersonGen(attempt_seed).generate(rid, level=job.get("level"), richness=richness)
            best = None
            shrinks = [-min(richness, 2), 0, 1, 2, 3] if target == 2 else [0, 1, 2, 3]
            for shrink in shrinks:
                style = make_style(tpl, person, attempt_seed, shrink)
                html = R.html(tpl, person, style)
                r = R.render(html)
                if not r["ok"]:
                    continue
                n = len(r["pages"])
                fill2 = page2_fill(r["pages"][1]) if n == 2 else 0
                if target == 1:
                    if n == 1:
                        best = (style, html, r, person)
                        break
                    best = best or (style, html, r, person)  # keep shrinking
                else:
                    if n == 2 and fill2 >= 0.25:
                        best = (style, html, r, person)
                        break
                    if n == 2:
                        best = best or (style, html, r, person)
                    if n == 1:
                        fallback = fallback or (style, html, r, person)  # use if 2 pages is unreachable
                        break  # needs more content, not smaller text
            if best and (target == 1 or (len(best[2]["pages"]) == 2 and page2_fill(best[2]["pages"][1]) >= 0.25)):
                chosen = best
                break
            chosen = chosen or best
        if chosen:
            break
        attempt_seed += 7777
    chosen = chosen or fallback
    if chosen is None:
        return dict(rid=rid, ok=False, reason="could not lay out")
    style, html, res, person = chosen
    res = R.render(html)  # re-render the chosen variant so the live page matches before screenshots
    if not res["ok"]:
        return dict(rid=rid, ok=False, reason=res.get("reason"))

    pages = res["pages"]
    mism = check_against_content(person, pages)
    n_pages = len(pages)
    for d in ("html", "ground_truth", "text"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for t in TIERS:
        for d in (f"images/{t}", f"annotations/{t}", f"yolo/labels/{t}"):
            (out / d).mkdir(parents=True, exist_ok=True)

    clean_paths = [out / "images" / "clean" / f"{rid}_p{i + 1}.png" for i in range(n_pages)]
    R.shoot(clean_paths)
    (out / "html" / f"{rid}.html").write_text(html, encoding="utf-8")

    gt = answer_key(person, pages)
    gt["_meta"] = dict(layout=TEMPLATE_LAYOUT[tpl], template=tpl, n_pages=n_pages, industry=person["industry"], level=person["level"],
                       years_experience=person["years_experience"], origin_country=person["origin_country"], work_country=person["work_country"],
                       date_format=person["date_format"])
    (out / "ground_truth" / f"{rid}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    full_text = []
    rows = []
    split = split_of(rid, tpl)
    for pi, pdata in enumerate(pages):
        ann, text = page_annotation(pdata)
        full_text.append(text)
        (out / "text" / f"{rid}_p{pi + 1}.txt").write_text(text + "\n", encoding="utf-8")
        clean = cv2.imread(str(clean_paths[pi]))
        for tier in TIERS:
            name = f"{rid}_p{pi + 1}"
            dseed = zlib.crc32(f"{name}-{tier}".encode()) ^ job["seed"]
            ext = "png" if (tier == "clean" or DEGRADED_PNG) else "jpg"
            if tier == "clean":
                img, H, params = clean, None, {"tier": "clean"}
            else:
                img, H, params = degrade(clean, tier, dseed)
                jb = params.pop("_jpeg_bytes")
                if ext == "jpg":
                    (out / "images" / tier / f"{name}.jpg").write_bytes(jb)  # last degradation step IS this JPEG
                else:
                    cv2.imwrite(str(out / "images" / tier / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            ih, iw = img.shape[:2]
            words, lines, regions = [], [], []
            for w in ann["words"]:
                poly, bb = transform_box(H, w["bbox"])
                words.append(dict(w, bbox=bb, poly=poly))
            for L in ann["lines"]:
                poly, bb = transform_box(H, L["bbox"])
                lines.append(dict(L, bbox=bb, poly=poly))
            yolo = []
            for r in ann["regions"]:
                poly, bb = transform_box(H, r["bbox"])
                cls = region_class(r["cls"])
                regions.append(dict(cls=cls, source=r["cls"], bbox=bb, poly=poly))
                x0, y0, x1, y1 = max(0, bb[0]), max(0, bb[1]), min(iw, bb[2]), min(ih, bb[3])
                if x1 - x0 > 2 and y1 - y0 > 2:
                    yolo.append(f"{YOLO_CLASSES.index(cls)} {(x0 + x1) / 2 / iw:.6f} {(y0 + y1) / 2 / ih:.6f} {(x1 - x0) / iw:.6f} {(y1 - y0) / ih:.6f}")
            doc = dict(resume_id=rid, page=pi + 1, n_pages=n_pages, tier=tier, image=f"images/{tier}/{name}.{ext}", width=iw, height=ih,
                       dpi=300, degradation=params, homography=None if H is None else [[round(float(v), 8) for v in row] for row in H],
                       words=words, lines=lines, regions=regions)
            (out / "annotations" / tier / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            (out / "yolo" / "labels" / tier / f"{name}.txt").write_text("\n".join(yolo) + "\n", encoding="utf-8")
            rows.append(dict(image=f"images/{tier}/{name}.{ext}", resume_id=rid, page=pi + 1, n_pages=n_pages, tier=tier,
                             layout=TEMPLATE_LAYOUT[tpl], template=tpl, split=split, industry=person["industry"], level=person["level"],
                             years_experience=person["years_experience"], origin_country=person["origin_country"], work_country=person["work_country"],
                             page_size=person["page_size"], width=iw, height=ih, n_words=len(words), font=style["font"].split(",")[0].strip("'"),
                             font_px=style["fs"], ground_truth=f"ground_truth/{rid}.json", annotation=f"annotations/{tier}/{name}.json",
                             gold_text=f"text/{name}.txt", degradation=json.dumps(params, ensure_ascii=False)))
    (out / "text" / f"{rid}.txt").write_text("\n\n".join(full_text) + "\n", encoding="utf-8")
    return dict(rid=rid, ok=True, rows=rows, mismatches=mism, pages=n_pages)


def _worker(args):
    job, out, dpng = args
    global DEGRADED_PNG
    DEGRADED_PNG = dpng
    try:
        return make_one(job, out)
    except Exception as e:  # keep going; report at the end
        import traceback
        return dict(rid=job["rid"], ok=False, reason=f"{type(e).__name__}: {e}", tb=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of unique resumes")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--degraded-png", action="store_true",
                    help="store mild/heavy pages as PNG instead of the JPEG produced by their final compression step (~15x larger)")
    ap.add_argument("--only", default=None, help="comma-separated resume ids to (re)build, e.g. R0001,R0007")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan(a.n, a.seed)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j["rid"] in keep]
    t0 = time.time()
    results = []
    if a.workers <= 1:
        for k, j in enumerate(jobs):
            results.append(_worker((j, out, a.degraded_png)))
            print(f"[{k + 1}/{len(jobs)}] {j['rid']} {j['template']} ok={results[-1]['ok']}", flush=True)
    else:
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_worker, [(j, out, a.degraded_png) for j in jobs])):
                results.append(r)
                print(f"[{k + 1}/{len(jobs)}] {r['rid']} ok={r['ok']}", flush=True)
    rows = [row for r in results if r["ok"] for row in r["rows"]]
    rows.sort(key=lambda r: (r["resume_id"], r["page"], list(TIERS).index(r["tier"])))
    mpath = out / "manifest.csv"
    if a.only and mpath.exists():  # merge into an existing manifest
        with open(mpath, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["resume_id"] not in {x["resume_id"] for x in rows}]
        rows = sorted(old + rows, key=lambda r: (r["resume_id"], int(r["page"]), list(TIERS).index(r["tier"])))
    if rows:
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "yolo" / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "yolo" / "classes.txt").write_text("\n".join(YOLO_CLASSES) + "\n")
    fails = [r for r in results if not r["ok"]]
    mism = [(r["rid"], r["mismatches"]) for r in results if r["ok"] and r["mismatches"]]
    print(f"\nDone in {time.time() - t0:.0f}s: {len(results) - len(fails)} resumes ok, {len(fails)} failed, {len(rows)} images.")
    for r in fails:
        print("FAILED", r["rid"], r.get("reason"))
    for rid, m in mism:
        print("FIELD MISMATCH", rid, m[:3])


if __name__ == "__main__":
    main()
