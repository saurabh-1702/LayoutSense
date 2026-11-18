"""Build every figure and table for the report from the saved results.

    python make_figures.py --data dataset --preds preds --ocr ocr --out report_figures\\figures

Reads   preds\\nlp_only, preds\\text_only (optional), preds\\layout_aware, preds\\label_ceiling (optional),
        ocr\\ocr_eval_summary.csv, report_figures\\training_log*.csv (optional)
Writes  (into --out)
  T1_main_results.csv / .md        mean per-resume F1 for every layout x tier x model
  T2_field_f1.csv                  per-field F1 for every layout x tier x model
  T3_significance.csv / .md        Wilcoxon signed-rank for every pair of models, Bonferroni-flagged
  T4_latency.csv / .md             parsing time per resume
  F1_heatmaps.png                  layout x tier F1 heatmap per model + difference maps
  F2_model_bars_clean.png          the three models (+ ceiling) per layout, clean tier
  F3_degradation_lines.png         F1 vs quality tier, one panel per layout
  F4_field_breakdown_multicol.png  per-field F1 on multi-column resumes (clean)
  F5_training_curves.png           validation F1 per epoch (if training logs exist)
  case1..case4_*.png               colour-coded side-by-side comparison tables (spec 11.1/11.2)
  overlays/*.png                   predicted field boxes vs answer key on the page image (spec 10.2)
  error_gallery/*.png              worst multi-column / heavy cases with what each model saw
"""
import argparse
import csv
import json
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import FIELDS, LISTS, SCALARS, prf, score_resume, sim  # noqa: E402

LAYOUTS = ["single_column", "multi_column", "table_based", "infographic"]
TIERS = ["clean", "mild", "heavy"]
MODEL_ORDER = ["nlp_only", "text_only", "layout_aware", "label_ceiling"]
NICE = {"nlp_only": "NLP-only (rules)", "text_only": "LayoutLMv3 text-only", "layout_aware": "LayoutLMv3 layout-aware",
        "label_ceiling": "Ceiling (perfect labels)"}
COLORS = {"nlp_only": "#c0504d", "text_only": "#e3a33b", "layout_aware": "#2f6fb0", "label_ceiling": "#9a9a9a"}
LAY_NICE = {"single_column": "Single-column", "multi_column": "Multi-column", "table_based": "Table-based", "infographic": "Infographic"}
GREEN, YELLOW, RED = "#c6efce", "#ffeb9c", "#ffc7ce"


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(path, rows, title):
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [f"### {title}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- scoring
def collect(data, preds_dir, split):
    resumes = {}
    for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8")):
        if r["page"] == "1" and r["split"] == split:
            resumes[(r["resume_id"], r["tier"])] = r
    models = [m for m in MODEL_ORDER if (preds_dir / m).exists()]
    per = {m: {} for m in models}          # (rid, tier) -> (resume F1, field scores, pred)
    for (rid, tier), r in resumes.items():
        gt = load_json(data / "ground_truth" / f"{rid}.json")
        for m in models:
            pf = preds_dir / m / tier / f"{rid}.json"
            pred = load_json(pf) if pf.exists() else {}
            sc = score_resume(gt, pred)
            tp = sum(v["tp"] for v in sc.values() if v)
            fp = sum(v["fp"] for v in sc.values() if v)
            fn = sum(v["fn"] for v in sc.values() if v)
            per[m][(rid, tier)] = (prf(tp, fp, fn)[2], sc, pred)
    return resumes, models, per


def cell_keys(resumes, layout, tier):
    return sorted(k for k, r in resumes.items() if r["layout"] == layout and k[1] == tier)


# ----------------------------------------------------------------------------- tables
def tables(resumes, models, per, out):
    t1, t2 = [], []
    for lay in LAYOUTS:
        for tier in TIERS:
            keys = cell_keys(resumes, lay, tier)
            if not keys:
                continue
            row = {"layout": lay, "tier": tier, "n": len(keys)}
            for m in models:
                row[m] = round(statistics.mean(per[m][k][0] for k in keys), 3)
            t1.append(row)
            for m in models:
                frow = {"layout": lay, "tier": tier, "model": m}
                for f in FIELDS:
                    tp = fp = fn = 0
                    for k in keys:
                        v = per[m][k][1].get(f)
                        if v:
                            tp, fp, fn = tp + v["tp"], fp + v["fp"], fn + v["fn"]
                    frow[f] = round(prf(tp, fp, fn)[2], 3) if tp + fp + fn else ""
                t2.append(frow)
    write_csv(out / "T1_main_results.csv", t1)
    write_md(out / "T1_main_results.md", t1, "Mean per-resume F1 (test split, held-out templates)")
    write_csv(out / "T2_field_f1.csv", t2)
    # significance
    pairs = [(a, b) for a, b in [("nlp_only", "layout_aware"), ("text_only", "layout_aware"), ("nlp_only", "text_only")] if a in models and b in models]
    n_tests = len(LAYOUTS) * len(TIERS)  # Bonferroni within each comparison family (12 cells)
    t3 = []
    for a, b in pairs:
        for lay in LAYOUTS:
            for tier in TIERS:
                keys = cell_keys(resumes, lay, tier)
                if not keys:
                    continue
                xa = [per[a][k][0] for k in keys]
                xb = [per[b][k][0] for k in keys]
                try:
                    p = wilcoxon(xa, xb).pvalue if any(x != y for x, y in zip(xa, xb)) else 1.0
                except ValueError:
                    p = 1.0
                t3.append(dict(comparison=f"{a} vs {b}", layout=lay, tier=tier, n=len(keys), mean_a=round(statistics.mean(xa), 3),
                               mean_b=round(statistics.mean(xb), 3), gain=round(statistics.mean(xb) - statistics.mean(xa), 3),
                               p_value=f"{p:.2e}", significant_bonferroni="yes" if p < 0.05 / n_tests else "no"))
    write_csv(out / "T3_significance.csv", t3)
    write_md(out / "T3_significance.md", t3, f"Wilcoxon signed-rank on paired per-resume F1 (Bonferroni: p < 0.05/{n_tests} = {0.05 / n_tests:.4f})")
    # latency
    t4 = []
    for m in models:
        if m == "label_ceiling":
            continue
        rts = [per[m][k][2].get("_runtime_s") for k in per[m] if per[m][k][2].get("_runtime_s") is not None]
        if rts:
            t4.append(dict(model=m, resumes=len(rts), mean_s=round(statistics.mean(rts), 3), median_s=round(statistics.median(rts), 3),
                           p95_s=round(sorted(rts)[int(0.95 * (len(rts) - 1))], 3),
                           note="CPU, after shared OCR" if m == "nlp_only" else "Kaggle T4 GPU, after shared OCR"))
    write_csv(out / "T4_latency.csv", t4)
    write_md(out / "T4_latency.md", t4, "Parsing time per resume (OCR excluded; OCR is shared by all models)")
    return t1, t2, t3


# ----------------------------------------------------------------------------- figures
def fig_heatmaps(t1, models, out):
    shown = [m for m in ["nlp_only", "text_only", "layout_aware"] if m in models]
    diffs = [(a, b) for a, b in [("nlp_only", "layout_aware"), ("text_only", "layout_aware")] if a in models and b in models]
    n = len(shown) + len(diffs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    lookup = {(r["layout"], r["tier"]): r for r in t1}
    for ax, m in zip(axes, shown):
        M = np.array([[lookup.get((l, t), {}).get(m, np.nan) for t in TIERS] for l in LAYOUTS], float)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_title(NICE[m], fontsize=10)
        _annot(ax, M)
    for ax, (a, b) in zip(axes[len(shown):], diffs):
        M = np.array([[lookup.get((l, t), {}).get(b, np.nan) - lookup.get((l, t), {}).get(a, np.nan) for t in TIERS] for l in LAYOUTS], float)
        ax.imshow(M, vmin=-0.6, vmax=0.6, cmap="RdBu")
        ax.set_title({"nlp_only": "Gain over NLP-only", "text_only": "Gain over text-only"}[a] + "\n(layout-aware minus it)", fontsize=10)
        _annot(ax, M, signed=True)
    for ax in axes:
        ax.set_xticks(range(3), [t.title() for t in TIERS])
        ax.set_yticks(range(4), [LAY_NICE[l] for l in LAYOUTS])
    fig.suptitle("Mean per-resume F1 by layout × image quality (test: held-out templates)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "F1_heatmaps.png", dpi=200)
    plt.close(fig)


def _annot(ax, M, signed=False):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}" if signed else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)


def fig_bars(t1, models, out, tier="clean"):
    lookup = {(r["layout"], r["tier"]): r for r in t1}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [lookup.get((l, tier), {}).get(m, 0) for l in LAYOUTS]
        b = ax.bar(np.arange(4) + (i - (len(models) - 1) / 2) * w, vals, w, label=NICE[m], color=COLORS[m],
                   hatch="//" if m == "label_ceiling" else None, alpha=0.6 if m == "label_ceiling" else 1)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(4), [LAY_NICE[l] for l in LAYOUTS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean per-resume F1")
    ax.set_title(f"Parser accuracy by layout ({tier} images, held-out templates)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"F2_model_bars_{tier}.png", dpi=200)
    plt.close(fig)


def fig_degradation(t1, models, out):
    lookup = {(r["layout"], r["tier"]): r for r in t1}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, lay in zip(axes, LAYOUTS):
        for m in models:
            ys = [lookup.get((lay, t), {}).get(m, np.nan) for t in TIERS]
            ax.plot(TIERS, ys, marker="o", color=COLORS[m], label=NICE[m], linestyle="--" if m == "label_ceiling" else "-")
        ax.set_title(LAY_NICE[lay])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean per-resume F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8)
    fig.suptitle("Effect of image degradation", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "F3_degradation_lines.png", dpi=200)
    plt.close(fig)


def fig_fields(t2, models, out, layout="multi_column", tier="clean"):
    rows = {r["model"]: r for r in t2 if r["layout"] == layout and r["tier"] == tier}
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [rows[m][f] if rows[m][f] != "" else 0 for f in FIELDS]
        ax.bar(np.arange(len(FIELDS)) + (i - (len(models) - 1) / 2) * w, vals, w, label=NICE[m], color=COLORS[m],
               alpha=0.6 if m == "label_ceiling" else 1)
    ax.set_xticks(range(len(FIELDS)), FIELDS, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-field F1 — {LAY_NICE[layout]} resumes, {tier} images")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "F4_field_breakdown_multicol.png", dpi=200)
    plt.close(fig)


def fig_training(logs_dir, out):
    logs = {}
    for p in Path(logs_dir).glob("training_log*.csv"):
        name = "layout_aware" if p.stem in ("training_log", "training_log_layout_aware") else ("text_only" if "text_only" in p.stem else p.stem)
        try:
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            logs[name] = rows
        except Exception:
            pass
    if not logs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, rows in logs.items():
        ep = [int(r["epoch"]) for r in rows]
        a1.plot(ep, [float(r["train_loss"]) for r in rows], marker="o", label=NICE.get(name, name), color=COLORS.get(name))
        a2.plot(ep, [float(r["val_token_f1"]) for r in rows], marker="o", label=NICE.get(name, name), color=COLORS.get(name))
    a1.set_title("Training loss"); a2.set_title("Validation token F1 (seen templates)")
    for a in (a1, a2):
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F5_training_curves.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- case tables
def _fmt_entries(entries, keys, n=2):
    out = []
    for e in (entries or [])[:n]:
        out.append(" @ ".join(str(e.get(k, "")) for k in keys if e.get(k)))
    return out


def field_rows(gt, preds):
    """[(field label, gt text, {model: (text, ratio)})]"""
    rows = []
    for f, label in [("name", "Name"), ("email", "Email"), ("phone", "Phone"), ("location", "Location")]:
        g = SCALARS[f](gt)
        if g:
            rows.append((label, g, {m: (SCALARS[f](p) or "—", sim(g, SCALARS[f](p), f) if SCALARS[f](p) else 0) for m, p in preds.items()}))
    for key, label, ks in [("experience", "Experience", ("title", "company")), ("education", "Education", ("degree", "institution"))]:
        gl = _fmt_entries(gt.get(key), ks)
        for i, gtxt in enumerate(gl):
            cells = {}
            for m, p in preds.items():
                pl = _fmt_entries(p.get(key), ks, n=20)
                best = max(pl, key=lambda x: sim(gtxt.replace(" @ ", " | "), x.replace(" @ ", " | ")), default="")
                cells[m] = (best or "—", sim(gtxt.replace(" @ ", " | "), best.replace(" @ ", " | ")) if best else 0)
            rows.append((f"{label} [{i + 1}]", gtxt, cells))
    gsk = gt.get("skills", [])
    if gsk:
        cells = {}
        for m, p in preds.items():
            psk = p.get("skills", [])
            hit = sum(1 for s in gsk if any(sim(s, x) >= 90 for x in psk))
            prec = sum(1 for x in psk if any(sim(s, x) >= 90 for s in gsk)) / max(1, len(psk))
            rec = hit / len(gsk)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            cells[m] = (f"{hit}/{len(gsk)} found; " + ", ".join(psk[:4]) + ("…" if len(psk) > 4 else ""), 100 * f1)
        rows.append(("Skills", ", ".join(gsk[:5]) + ("…" if len(gsk) > 5 else ""), cells))
    return rows


def case_table(title, rid, tier, layout, gt, preds, models, out_path):
    rows = field_rows(gt, {m: preds[m] for m in models})
    cols = ["Field", "Ground truth"] + [NICE[m] for m in models]
    wrap = lambda s, w=26: "\n".join(textwrap.wrap(str(s), w)) or "—"
    cell_text, cell_col = [], []
    for label, g, cells in rows:
        cell_text.append([label, wrap(g)] + [wrap(cells[m][0]) for m in models])
        cell_col.append(["#f2f2f2", "#ffffff"] + [GREEN if cells[m][1] >= 90 else YELLOW if cells[m][1] >= 50 else RED for m in models])
    line_counts = [max(c.count("\n") + 1 for c in r) for r in cell_text]
    total = sum(line_counts) + 1.2
    h = 0.9 + 0.24 * total
    fig, ax = plt.subplots(figsize=(3.0 + 2.6 * (len(cols) - 1), h))
    ax.axis("off")
    tb = ax.table(cellText=cell_text, cellColours=cell_col, colLabels=cols, cellLoc="left", bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#404040"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        cell.set_height((1.2 if r == 0 else line_counts[r - 1]) / total)
    ax.set_title(f"{title}\n{rid} · {LAY_NICE[layout]} · {tier} images   (green ≥ 90% match, yellow 50–89%, red < 50% or missing)",
                 fontsize=9, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pick_representative(resumes, per, layout, tier, models):
    """Resume whose layout-aware minus NLP gap is closest to the cell MEDIAN gap (not the best case)."""
    keys = cell_keys(resumes, layout, tier)
    if not keys or "nlp_only" not in models or "layout_aware" not in models:
        return None
    gaps = {k: per["layout_aware"][k][0] - per["nlp_only"][k][0] for k in keys}
    med = statistics.median(gaps.values())
    return min(keys, key=lambda k: abs(gaps[k] - med))


# ----------------------------------------------------------------------------- overlays
FIELD_COLORS = {}


def _col(t):
    if t not in FIELD_COLORS:
        import zlib
        h = zlib.crc32(t.encode()) % 360
        FIELD_COLORS[t] = matplotlib.colors.hsv_to_rgb((h / 360, 0.65, 0.95))
    return FIELD_COLORS[t]


def overlay(data, preds_dir, resumes, rid, tier, out_path, model="layout_aware"):
    tokp = preds_dir / model / "_tokens" / tier / f"{rid}.json"
    r = next((x for x in csv.DictReader(open(data / "manifest.csv", encoding="utf-8"))
              if x["resume_id"] == rid and x["tier"] == tier and x["page"] == "1"), None)
    if not tokp.exists() or r is None:
        return False
    toks = load_json(tokp)[0]
    ann = load_json(data / r["annotation"])
    img = plt.imread(str(data / r["image"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.2))
    for ax, title, items in [(axes[0], "Answer key (true fields)", [(w["bbox"], w["label"].split("-", 1)[-1]) for w in ann["words"]]),
                             (axes[1], f"{NICE[model]} prediction", list(zip(toks["orig_boxes"], [l.split("-", 1)[-1] for l in toks["pred"]])))]:
        ax.imshow(img)
        for b, t in items:
            if t == "O" or t == "SECTION_HEADING":
                continue
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=True, alpha=0.35, color=_col(t), lw=0))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    types = sorted({w["label"].split("-", 1)[-1] for w in ann["words"]} | {l.split("-", 1)[-1] for l in toks["pred"]})
    types = [t for t in types if t not in ("O", "SECTION_HEADING")]
    handles = [plt.Rectangle((0, 0), 1, 1, color=_col(t), alpha=0.6) for t in types]
    fig.legend(handles, types, loc="lower center", ncol=min(8, len(types)), fontsize=7)
    fig.suptitle(f"{rid} · {LAY_NICE[r['layout']]} · {tier}", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def gallery_item(data, preds_dir, resumes, per, models, rid, tier, out_path):
    r = resumes[(rid, tier)]
    img = plt.imread(str(data / next(x["image"] for x in csv.DictReader(open(data / "manifest.csv", encoding="utf-8"))
                                     if x["resume_id"] == rid and x["tier"] == tier and x["page"] == "1")))
    flat = preds_dir / "nlp_only" / "_flat" / tier / f"{rid}.txt"
    flat_txt = flat.read_text(encoding="utf-8").splitlines()[:38] if flat.exists() else []
    gt = load_json(data / "ground_truth" / f"{rid}.json")
    fig = plt.figure(figsize=(15, 8))
    a1 = fig.add_axes([0.0, 0.0, 0.34, 0.93]); a1.imshow(img); a1.axis("off"); a1.set_title("Page image", fontsize=9)
    a2 = fig.add_axes([0.35, 0.0, 0.32, 0.93]); a2.axis("off")
    a2.set_title("What the NLP-only model read (raster-flattened OCR)", fontsize=9)
    a2.text(0, 1, "\n".join(textwrap.shorten(l, 70) for l in flat_txt), va="top", family="monospace", fontsize=6.5)
    a3 = fig.add_axes([0.68, 0.0, 0.32, 0.93]); a3.axis("off"); a3.set_title("Experience entries: truth vs models", fontsize=9)
    lines = ["TRUTH:"] + [f"  {e.get('title', '')} @ {e.get('company', '')}" for e in gt.get("experience", [])[:4]]
    for m in models:
        if m == "label_ceiling":
            continue
        p = per[m][(rid, tier)][2]
        lines += ["", f"{NICE[m]}  (resume F1 {per[m][(rid, tier)][0]:.2f}):"]
        lines += [f"  {e.get('title') or '—'} @ {e.get('company') or '—'}" for e in p.get("experience", [])[:4]] or ["  (no experience found)"]
    a3.text(0, 1, "\n".join(textwrap.shorten(l, 60) for l in lines), va="top", family="monospace", fontsize=7)
    fig.suptitle(f"{rid} · {LAY_NICE[r['layout']]} · {tier}", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--ocr", default="ocr")
    ap.add_argument("--logs", default="report_figures")
    ap.add_argument("--out", default="report_figures/figures")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    data, preds_dir, out = Path(a.data), Path(a.preds), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resumes, models, per = collect(data, preds_dir, a.split)
    print("models found:", models)
    t1, t2, t3 = tables(resumes, models, per, out)
    fig_heatmaps(t1, models, out)
    for tier in TIERS:
        fig_bars(t1, models, out, tier)
    fig_degradation(t1, models, out)
    fig_fields(t2, models, out)
    fig_training(a.logs, out)
    print("tables + charts done")

    # four case scenarios (spec 11.2) - representative = closest to the median gap of that cell
    main_models = [m for m in ["nlp_only", "text_only", "layout_aware"] if m in models]
    cases = [("Case 1 - Clean single-column (parity baseline)", "single_column", "clean"),
             ("Case 2 - Multi-column, clean (layout variable isolated)", "multi_column", "clean"),
             ("Case 3 - Single-column, heavy degradation (quality variable isolated)", "single_column", "heavy"),
             ("Case 4 - Multi-column, heavy degradation (combined worst case)", "multi_column", "heavy"),
             ("Case 4b - Table-based, heavy degradation", "table_based", "heavy")]
    (out / "overlays").mkdir(exist_ok=True)
    sel = []
    for i, (title, lay, tier) in enumerate(cases, 1):
        k = pick_representative(resumes, per, lay, tier, models)
        if not k:
            continue
        rid = k[0]
        gt = load_json(data / "ground_truth" / f"{rid}.json")
        preds = {m: per[m][k][2] for m in main_models}
        tag = title.split(" - ")[0].lower().replace(" ", "")
        case_table(title, rid, tier, lay, gt, preds, main_models, out / f"{tag}_{lay}_{tier}_{rid}.png")
        overlay(data, preds_dir, resumes, rid, tier, out / "overlays" / f"{tag}_{rid}_{tier}_layout_aware.png")
        sel.append(dict(case=title, resume_id=rid, layout=lay, tier=tier, **{f"F1_{m}": round(per[m][k][0], 3) for m in main_models}))
    write_csv(out / "case_selection.csv", sel)
    # one overlay per layout (clean), layout-aware model
    for lay in LAYOUTS:
        k = pick_representative(resumes, per, lay, "clean", models)
        if k:
            overlay(data, preds_dir, resumes, k[0], "clean", out / "overlays" / f"{lay}_{k[0]}_clean_layout_aware.png")
            if "text_only" in models:
                overlay(data, preds_dir, resumes, k[0], "clean", out / "overlays" / f"{lay}_{k[0]}_clean_text_only.png", model="text_only")
    print("case tables + overlays done")

    # error gallery: largest NLP failures on multi-column, and the hardest heavy pages for the layout-aware model
    (out / "error_gallery").mkdir(exist_ok=True)
    picks = []
    if "nlp_only" in models and "layout_aware" in models:
        mc = cell_keys(resumes, "multi_column", "clean") + cell_keys(resumes, "multi_column", "mild")
        picks += sorted(mc, key=lambda k: per["nlp_only"][k][0] - per["layout_aware"][k][0])[:3]
        hv = [k for k in per["layout_aware"] if k[1] == "heavy"]
        picks += sorted(hv, key=lambda k: per["layout_aware"][k][0])[:3]
    for rid, tier in picks:
        gallery_item(data, preds_dir, resumes, per, models, rid, tier, out / "error_gallery" / f"{rid}_{tier}.png")
    print(f"error gallery: {len(picks)} items\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
