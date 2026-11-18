"""Audit the whole project on your PC: checks that every file exists and is complete, and recomputes the
paper's key numbers from your prediction files.

    python tools\\audit_project.py

Writes report_figures\\audit_report.txt (send that file back). Read-only: it changes nothing.
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import prf, score_resume  # noqa: E402

LAYOUTS = ["single_column", "multi_column", "table_based", "infographic"]
TIERS = ["clean", "mild", "heavy"]
MODELS = ["nlp_only", "text_only", "layout_aware", "label_ceiling"]
# numbers printed in the paper (mean per-resume F1, n = 25 per cell)
PAPER = {
    "nlp_only": [0.846, 0.831, 0.457, 0.346, 0.382, 0.192, 0.441, 0.397, 0.154, 0.303, 0.291, 0.236],
    "text_only": [0.947, 0.933, 0.560, 0.756, 0.754, 0.496, 0.527, 0.502, 0.199, 0.738, 0.733, 0.559],
    "layout_aware": [0.966, 0.957, 0.569, 0.936, 0.914, 0.603, 0.644, 0.581, 0.216, 0.769, 0.744, 0.592],
}
PAPER_P = {  # Wilcoxon p-values in the paper, same cell order
    ("nlp_only", "layout_aware"): [2.702e-05, 2.702e-05, 0.0005167, 1.228e-05, 5.96e-08, 8.857e-05, 8.345e-07, 1.132e-06, 0.001696, 5.96e-08, 5.96e-08, 2.702e-05],
    ("text_only", "layout_aware"): [0.04139, 0.03033, 0.4769, 1.82e-05, 3.088e-05, 0.0004552, 4.968e-05, 0.001069, 0.4688, 0.2304, 0.5872, 0.07772],
    ("nlp_only", "text_only"): [6.766e-05, 4.021e-05, 0.0006806, 5.96e-08, 5.96e-08, 0.0001033, 0.01472, 0.001296, 0.008968, 5.96e-08, 5.96e-08, 2.702e-05],
}
CELLS = [(l, t) for l in LAYOUTS for t in TIERS]
out_lines, problems, warnings = [], [], []


def say(s=""):
    print(s)
    out_lines.append(s)


def problem(s):
    problems.append(s)
    say("  [PROBLEM] " + s)


def warn(s):
    warnings.append(s)
    say("  [warning] " + s)


def ok(s):
    say("  [ok] " + s)


def main():
    root = Path(".")
    data = root / "dataset"
    say("=== 1. Dataset ===")
    if not (data / "manifest.csv").exists():
        problem("dataset\\manifest.csv not found - run this from D:\\projectss\\resume_synth")
        return finish()
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    rids = sorted({r["resume_id"] for r in rows})
    per_tier = Counter(r["tier"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    two_page = sum(1 for r in rows if r["tier"] == "clean" and r["page"] == "2")
    test_by_layout = Counter(r["layout"] for r in rows if r["split"] == "test" and r["tier"] == "clean" and r["page"] == "1")
    say(f"  resumes {len(rids)}, page images {len(rows)}, per tier {dict(per_tier)}, two-page resumes {two_page}")
    say(f"  pages by split (all tiers) {dict(splits)}; test resumes per layout {dict(test_by_layout)}")
    expect = dict(resumes=300, images=1278, two_page=126, train=714, val=123, test=441)
    got = dict(resumes=len(rids), images=len(rows), two_page=two_page, train=splits["train"], val=splits["val"], test=splits["test"])
    for k in expect:
        (ok if expect[k] == got[k] else problem)(f"{k}: {got[k]} (paper says {expect[k]})")
    miss_img = [r["image"] for r in rows if not (data / r["image"]).exists()]
    miss_ann = [r["annotation"] for r in rows if not (data / r["annotation"]).exists()]
    miss_gt = [rid for rid in rids if not (data / "ground_truth" / f"{rid}.json").exists()]
    for name, m in (("images", miss_img), ("annotations", miss_ann), ("ground-truth files", miss_gt)):
        (ok if not m else problem)(f"missing {name}: {len(m)}" + (f" e.g. {m[:3]}" if m else ""))

    say("\n=== 2. OCR ===")
    ocr_dir = root / "ocr" / "tesseract"
    n_ocr = {t: len(list((ocr_dir / t).glob("*.json"))) for t in TIERS}
    for t in TIERS:
        (ok if n_ocr[t] == per_tier[t] else problem)(f"OCR files {t}: {n_ocr[t]} / {per_tier[t]} pages")
    summ = root / "ocr" / "ocr_eval_summary.csv"
    if summ.exists():
        say("  ocr_eval_summary.csv:")
        for line in summ.read_text(encoding="utf-8").splitlines():
            say("    " + line)
    else:
        warn("ocr\\ocr_eval_summary.csv not found (needed for the OCR table) - run pipeline\\eval_ocr.py")

    say("\n=== 3. Predictions (test split) ===")
    test = {(r["resume_id"], r["tier"]): r for r in rows if r["split"] == "test" and r["page"] == "1"}
    gts = {rid: json.loads((data / "ground_truth" / f"{rid}.json").read_text(encoding="utf-8")) for rid in {k[0] for k in test}}
    per = {}
    runtimes = {}
    for m in MODELS:
        d = root / "preds" / m
        if not d.exists():
            (warn if m == "label_ceiling" else problem)(f"preds\\{m} not found")
            continue
        missing = [k for k in test if not (d / k[1] / f"{k[0]}.json").exists()]
        (ok if not missing else problem)(f"preds\\{m}: {len(test) - len(missing)}/{len(test)} test predictions" + (f", missing e.g. {missing[:3]}" if missing else ""))
        per[m] = {}
        rts = []
        for k in test:
            pf = d / k[1] / f"{k[0]}.json"
            pred = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
            if pred.get("_runtime_s") is not None:
                rts.append(pred["_runtime_s"])
            sc = score_resume(gts[k[0]], pred)
            tp = sum(v["tp"] for v in sc.values() if v)
            fp = sum(v["fp"] for v in sc.values() if v)
            fn = sum(v["fn"] for v in sc.values() if v)
            per[m][k] = prf(tp, fp, fn)[2]
        runtimes[m] = rts
        if m in ("text_only", "layout_aware") and not (d / "_tokens").exists():
            warn(f"preds\\{m}\\_tokens missing (needed for overlay figures)")

    say("\n=== 4. Recomputed mean per-resume F1 vs the paper ===")
    say(f"  {'layout':14} {'tier':6} " + " ".join(f"{m[:12]:>13}" for m in PAPER))
    worst = 0.0
    for i, (l, t) in enumerate(CELLS):
        keys = [k for k in test if test[k]["layout"] == l and k[1] == t]
        cells = []
        for m in PAPER:
            if m not in per:
                cells.append(f"{'-':>13}")
                continue
            v = statistics.mean(per[m][k] for k in keys)
            worst = max(worst, abs(v - PAPER[m][i]))
            cells.append(f"{v:.3f}({PAPER[m][i]:.3f})".rjust(13))
        say(f"  {l:14} {t:6} " + " ".join(cells))
    say("  (value recomputed now, paper value in brackets)")
    (ok if worst <= 0.0015 else problem)(f"largest difference from the paper: {worst:.4f}")

    try:
        from scipy.stats import wilcoxon
        say("\n=== 5. Recomputed Wilcoxon p-values vs the paper ===")
        for (a, b), pp in PAPER_P.items():
            if a not in per or b not in per:
                continue
            bad = 0
            for i, (l, t) in enumerate(CELLS):
                keys = sorted(k for k in test if test[k]["layout"] == l and k[1] == t)
                xa, xb = [per[a][k] for k in keys], [per[b][k] for k in keys]
                p = wilcoxon(xa, xb).pvalue if any(x != y for x, y in zip(xa, xb)) else 1.0
                if abs(p - pp[i]) / pp[i] > 0.01:
                    bad += 1
                    say(f"    {a} vs {b} {l} {t}: now {p:.3g}, paper {pp[i]:.3g}")
            (ok if not bad else problem)(f"{a} vs {b}: {12 - bad}/12 p-values match")
    except ImportError:
        warn("scipy not installed - p-values not rechecked")

    say("\n=== 6. Parsing time per resume (from _runtime_s in the prediction files) ===")
    for m, rts in runtimes.items():
        if rts:
            say(f"  {m:14} mean {statistics.mean(rts):.3f} s, median {statistics.median(rts):.3f} s over {len(rts)} resumes")
        elif m != "label_ceiling":
            warn(f"{m}: no _runtime_s values")

    say("\n=== 7. Report files ===")
    rf = root / "report_figures"
    for f in ("training_log.csv", "training_log_text_only.csv"):
        p = rf / f
        if p.exists():
            last = list(csv.DictReader(open(p, encoding="utf-8")))[-1]
            ok(f"{f}: {last.get('epoch')} epochs, final val_token_f1 {last.get('val_token_f1')}")
        else:
            warn(f"report_figures\\{f} not found (training-curve figure)")
    figs = rf / "figures"
    n_figs = len(list(figs.rglob("*.png"))) if figs.exists() else 0
    (ok if n_figs else warn)(f"report_figures\\figures: {n_figs} PNG files")
    cnn = rf / "vision" / "cnn_test_report.txt"
    (ok if cnn.exists() else warn)("CNN report " + ("found" if cnn.exists() else "not found in report_figures\\vision"))

    say("\n=== 8. YOLO / vision upload ===")
    vd = root / "vision_data"
    if vd.exists():
        labs = list((vd / "yolo" / "labels").glob("*/*.txt"))
        empty = [p for p in labs if not p.read_text().strip()]
        imgs = list((vd / "yolo" / "images").glob("*/*.jpg"))
        (ok if len(imgs) == len(rows) and len(labs) == len(rows) and not empty else problem)(
            f"vision_data: {len(imgs)} images, {len(labs)} label files, {len(empty)} empty")
    else:
        warn("vision_data folder not found (needed for the YOLO re-run)")
    nb = root / "pipeline" / "train_vision_kaggle.ipynb"
    if nb.exists():
        txt = nb.read_text(encoding="utf-8")
        (ok if "check passed" in txt and "copytree" not in txt else problem)(
            "train_vision_kaggle.ipynb is the " + ("fixed version" if "check passed" in txt else "OLD version - replace it before the YOLO re-run"))
    return finish()


def finish():
    say(f"\n=== SUMMARY: {len(problems)} problem(s), {len(warnings)} warning(s) ===")
    for p in problems:
        say("  PROBLEM: " + p)
    for w in warnings:
        say("  warning: " + w)
    out = Path("report_figures")
    out.mkdir(exist_ok=True)
    (out / "audit_report.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {(out / 'audit_report.txt').resolve()}")


if __name__ == "__main__":
    main()
