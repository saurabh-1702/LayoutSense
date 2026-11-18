"""Diagnose Model B predictions against the true word labels, and re-decode them robustly.

    python pipeline\\diagnose_layoutlm.py --labels layoutlm_data --pred preds\\layout_aware --out preds\\layout_aware_v2

1. Compares the predicted label of every test word with its true label (from layoutlm_data/examples.jsonl):
   - field-type accuracy  (e.g. EXP_COMPANY, ignoring the B-/I- prefix)
   - B/I prefix accuracy  (does the model mark where a value starts?)
2. Re-decodes the saved word predictions into answer-key JSON in --out using TYPE-ONLY grouping:
   neighbouring words with the same field type form one value; a new value starts when the type
   changes or a new line begins far away. This does not depend on the model's B-/I- prefixes.
   Then score:  python score.py --data dataset --pred preds\\nlp_only --pred preds\\layout_aware_v2 --split test
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.layoutlm_lib import decode  # noqa: E402


def type_only_relabel(words, labels, boxes=None):
    """Rebuild B-/I- from field types alone. boxes = 0-1000 boxes on the straightened page.
    A word continues the previous value if it has the same type and is either right next to it on
    the same line (gap no wider than a normal space) or starts the following line (wrapped value)."""
    out, prev_t, prev_b = [], None, None
    for i, l in enumerate(labels):
        if l == "O":
            out.append("O")
            prev_t, prev_b = None, None
            continue
        t = l.split("-", 1)[1]
        b = boxes[i] if boxes else None
        cont = t == prev_t
        if cont and b and prev_b:
            h = max(1.0, prev_b[3] - prev_b[1])
            hx = h * 1.41  # page-normalised units: x is stretched vs y on A4/Letter
            same_line = abs((b[1] + b[3]) / 2 - (prev_b[1] + prev_b[3]) / 2) < 0.6 * h and -0.5 * hx < b[0] - prev_b[2] < 0.9 * hx
            next_line = 0 <= b[1] - prev_b[3] < 1.0 * h and b[0] < prev_b[2]
            cont = same_line or next_line
        out.append(("I-" if cont else "B-") + t)
        prev_t, prev_b = t, b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="layoutlm_data", help="folder with examples.jsonl")
    ap.add_argument("--pred", default="preds/layout_aware")
    ap.add_argument("--out", default="preds/layout_aware_v2")
    a = ap.parse_args()
    gold = {}
    for line in open(Path(a.labels) / "examples.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e["split"] == "test":
            gold[(e["resume_id"], e["tier"], e["page"])] = e
    tok_dir = Path(a.pred) / "_tokens"
    files = sorted(tok_dir.glob("*/*.json"))
    if not files:
        sys.exit(f"No token predictions in {tok_dir}")
    n = type_ok = full_ok = 0
    nonO = nonO_type_ok = 0
    prefix = Counter()
    pred_prefix = Counter()
    misaligned = 0
    out = Path(a.out)
    for f in files:
        tier, rid = f.parent.name, f.stem
        pages = json.loads(f.read_text(encoding="utf-8"))
        all_w, all_l = [], []
        for p in pages:
            g = gold.get((rid, tier, p["page"]))
            if g is None or g["words"] != p["words"]:
                misaligned += 1
            else:
                for gl, pl in zip(g["labels"], p["pred"]):
                    n += 1
                    gt, pt = gl.split("-")[-1], pl.split("-")[-1]
                    type_ok += gt == pt
                    full_ok += gl == pl
                    pred_prefix[pl.split("-")[0]] += 1
                    if gl != "O":
                        nonO += 1
                        nonO_type_ok += gt == pt
                        if gt == pt:
                            prefix[(gl[0], pl[0])] += 1
            all_w += p["words"]
            bx = g["boxes"] if (g is not None and g["words"] == p["words"]) else None
            all_l += type_only_relabel(p["words"], p["pred"], bx)
        (out / tier).mkdir(parents=True, exist_ok=True)
        (out / tier / f"{rid}.json").write_text(json.dumps({"resume_id": rid, **decode(all_w, all_l)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pages checked: {len(files)} resumes x tiers, {misaligned} pages could not be aligned to the true labels")
    if n:
        print(f"words: {n}")
        print(f"  label exactly right (incl. B/I):        {full_ok / n:.3f}")
        print(f"  field type right (ignoring B/I):        {type_ok / n:.3f}")
        print(f"  field type right on non-O words:        {nonO_type_ok / max(1, nonO):.3f}")
        tot = sum(prefix.values()) or 1
        print("  B/I on correctly-typed words (true -> predicted):",
              ", ".join(f"{k[0]}->{k[1]}: {v / tot:.2%}" for k, v in sorted(prefix.items())))
        print("  predicted prefixes overall:", dict(pred_prefix))
    print(f"\nRe-decoded predictions written to {out.resolve()}")


if __name__ == "__main__":
    main()
