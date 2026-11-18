"""Score model predictions against the answer key.

Prediction layout (one JSON per resume per tier, same schema as ground_truth/*.json;
fields a model does not produce can simply be omitted):

    preds/<model_name>/<tier>/<RESUME_ID>.json

Usage:
    python score.py --data dataset --pred preds/nlp_only --pred preds/layout_aware --split test

Writes <pred_dir>/scores_per_resume.csv and prints F1 per field for every
layout x tier cell, plus a Wilcoxon signed-rank test when two models are given.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

FUZZY_OK = 90  # spec 11.1: >= 90 green, 50-89 yellow, < 50 red

SCALARS = {
    "name": lambda d: d.get("name"),
    "email": lambda d: (d.get("contact") or {}).get("email"),
    "phone": lambda d: (d.get("contact") or {}).get("phone"),
    "location": lambda d: (d.get("contact") or {}).get("location"),
}
LISTS = {
    "education": lambda e: " | ".join(str(e.get(k, "")) for k in ("degree", "institution")),
    "experience": lambda e: " | ".join(str(e.get(k, "")) for k in ("title", "company")),
    "skills": lambda e: e if isinstance(e, str) else str(e),
    "certifications": lambda e: e.get("name", "") if isinstance(e, dict) else str(e),
    "languages": lambda e: e.get("language", "") if isinstance(e, dict) else str(e),
    "projects": lambda e: e.get("name", "") if isinstance(e, dict) else str(e),
}
FIELDS = list(SCALARS) + list(LISTS)


def norm(s, field=None):
    s = str(s or "").strip()
    if field == "phone":
        return re.sub(r"\D", "", s)[-9:]  # compare last 9 digits: ignores +CC / trunk 0 formatting
    if field == "email":
        return s.lower().replace(" ", "")
    return re.sub(r"\s+", " ", s).lower()


def sim(a, b, field=None):
    a, b = norm(a, field), norm(b, field)
    if not a and not b:
        return 100.0
    if field in ("phone", "email"):
        return 100.0 * Levenshtein.normalized_similarity(a, b)
    return float(fuzz.token_sort_ratio(a, b))


def score_scalar(gt, pr, field):
    if not gt:
        return None
    if not pr:
        return dict(tp=0, fp=0, fn=1, exact=0, ratio=0.0)
    r = sim(gt, pr, field)
    exact = int(norm(gt, field) == norm(pr, field))
    ok = r >= FUZZY_OK
    return dict(tp=int(ok), fp=int(not ok), fn=int(not ok), exact=exact, ratio=r)


def score_list(gt_items, pr_items, key):
    g = [key(x) for x in (gt_items or [])]
    p = [key(x) for x in (pr_items or [])]
    if not g:
        return None
    pairs = sorted(((sim(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
    used_g, used_p, tp, exact, ratios = set(), set(), 0, 0, []
    for r, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i), used_p.add(j)
        ratios.append(r)
        if r >= FUZZY_OK:
            tp += 1
            exact += int(norm(g[i]) == norm(p[j]))
    ratios += [0.0] * (len(g) - len(ratios))
    return dict(tp=tp, fp=len(p) - tp, fn=len(g) - tp, exact=exact, ratio=sum(ratios) / len(g))


def score_resume(gt, pred):
    out = {}
    for f, get in SCALARS.items():
        out[f] = score_scalar(get(gt), get(pred), f)
    for f, key in LISTS.items():
        out[f] = score_list(gt.get(f), pred.get(f), key)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--pred", action="append", required=True, help="prediction folder (repeat for 2 models)")
    ap.add_argument("--split", default="test", help="train / val / test / all")
    a = ap.parse_args()
    data = Path(a.data)
    resumes = {}
    with open(data / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["page"] == "1" and (a.split == "all" or r["split"] == a.split):
                resumes[(r["resume_id"], r["tier"])] = r["layout"]

    per_model_f1 = {}
    for pdir in map(Path, a.pred):
        rows, agg = [], defaultdict(lambda: [0, 0, 0])
        per_model_f1[pdir.name] = {}
        for (rid, tier), layout in sorted(resumes.items()):
            gt = json.loads((data / "ground_truth" / f"{rid}.json").read_text(encoding="utf-8"))
            pf = pdir / tier / f"{rid}.json"
            pred = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
            sc = score_resume(gt, pred)
            tp = sum(v["tp"] for v in sc.values() if v)
            fp = sum(v["fp"] for v in sc.values() if v)
            fn = sum(v["fn"] for v in sc.values() if v)
            per_model_f1[pdir.name][(rid, tier)] = prf(tp, fp, fn)[2]
            row = dict(resume_id=rid, tier=tier, layout=layout, missing_prediction=int(not pf.exists()))
            for fld, v in sc.items():
                if v is None:
                    continue
                row[f"{fld}_f1"] = round(prf(v["tp"], v["fp"], v["fn"])[2], 4)
                row[f"{fld}_ratio"] = round(v["ratio"], 1)
                for k, x in zip(("tp", "fp", "fn"), (v["tp"], v["fp"], v["fn"])):
                    agg[(layout, tier, fld)][("tp", "fp", "fn").index(k)] += x
            rows.append(row)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("resume_id", "tier", "layout", "missing_prediction"), k))
        with open(pdir / "scores_per_resume.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== {pdir.name}: F1 by layout x tier (fuzzy >= {FUZZY_OK}) ===")
        print(f"{'layout':15s} {'tier':6s} " + " ".join(f"{x[:8]:>8s}" for x in FIELDS))
        for layout in sorted({k[0] for k in agg}):
            for tier in ("clean", "mild", "heavy"):
                cells = [agg.get((layout, tier, fld)) for fld in FIELDS]
                if not any(cells):
                    continue
                print(f"{layout:15s} {tier:6s} " + " ".join(f"{prf(*c)[2]:8.3f}" if c else f"{'-':>8s}" for c in cells))

    if len(a.pred) == 2:
        from scipy.stats import wilcoxon
        m1, m2 = list(per_model_f1)
        print(f"\n=== Wilcoxon signed-rank on per-resume F1: {m1} vs {m2} ===")
        cells = defaultdict(list)
        for key in per_model_f1[m1]:
            cells[(resumes[key], key[1])].append((per_model_f1[m1][key], per_model_f1[m2][key]))
        for (layout, tier), xs in sorted(cells.items()):
            d = [x - y for x, y in xs]
            try:
                stat = wilcoxon([x for x, _ in xs], [y for _, y in xs]) if any(d) else None
                pv = f"{stat.pvalue:.4g}" if stat else "n/a (identical)"
            except ValueError:
                pv = "n/a"
            print(f"{layout:15s} {tier:6s} n={len(xs):3d}  mean F1 {sum(x for x, _ in xs) / len(xs):.3f} vs {sum(y for _, y in xs) / len(xs):.3f}  p={pv}")


if __name__ == "__main__":
    main()
