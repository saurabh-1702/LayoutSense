"""Live demo engine: runs ONE resume through every stage of the project and returns everything the UI shows.

Stages (all live on your PC):
  1. CV preprocessing (same code as the experiments)     5. CNN layout classifier (ResNet-18)
  2. Tesseract OCR (same settings)                        6. YOLO11 region detector
  3. NLP-only parser (raster flattening + rules + spaCy)
  4. LayoutLMv3 text-only and layout-aware - LIVE if their weights are in models\\, otherwise the
     RECORDED test-set output of the trained models is used for dataset resumes (clearly labelled).
Every number is computed with the project's own scoring code (score.py).
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("DEMO_ROOT", CODE))   # project folder (dataset, preds, models); defaults to the code folder
sys.path.insert(0, str(CODE))
os.environ.setdefault("OMP_THREAD_LIMIT", "4")

from pipeline.preprocess import preprocess  # noqa: E402
from pipeline.run_ocr import TESS_CONFIG, find_tesseract, map_back  # noqa: E402
from pipeline.layoutlm_lib import (ID2LABEL, decode, encode_window, make_windows, norm_box,  # noqa: E402
                                   predict_page, project_labels, raster_order)
from score import FIELDS, LISTS, SCALARS, prf, score_resume, sim  # noqa: E402

LAYOUTS = ["single_column", "multi_column", "table_based", "infographic"]
LAYOUT_NICE = {"single_column": "Single-column", "multi_column": "Multi-column", "table_based": "Table-based", "infographic": "Infographic"}
MODELS = ["nlp_only", "text_only", "layout_aware"]
MODEL_NICE = {"nlp_only": "NLP-only (rules)", "text_only": "LayoutLMv3 text-only", "layout_aware": "LayoutLMv3 layout-aware"}
YOLO_NAMES = ["header", "sidebar", "column", "table", "section_experience", "section_education", "section_skills",
              "section_projects", "section_certifications", "section_awards", "section_languages", "section_contact",
              "section_links", "section_other"]

# field groups used for colouring words and boxes (32 token types -> 10 readable groups)
GROUP_OF = {"NAME": "Name", "HEADLINE": "Name", "EMAIL": "Contact", "PHONE": "Contact", "LOCATION": "Contact", "LINK": "Contact",
            "EDU_DEGREE": "Education", "EDU_INSTITUTION": "Education", "EDU_DATE": "Education", "EDU_GRADE": "Education", "EDU_LOCATION": "Education",
            "EXP_TITLE": "Experience", "EXP_COMPANY": "Experience", "EXP_DATE": "Experience", "EXP_LOCATION": "Experience", "EXP_DESC": "Experience",
            "SKILL": "Skills", "SKILL_CATEGORY": "Skills", "SKILL_LEVEL": "Skills",
            "PROJ_NAME": "Projects", "PROJ_DESC": "Projects", "PROJ_TECH": "Projects", "PROJ_DATE": "Projects",
            "CERT_NAME": "Certs & awards", "CERT_ISSUER": "Certs & awards", "CERT_DATE": "Certs & awards",
            "AWARD_NAME": "Certs & awards", "AWARD_ISSUER": "Certs & awards", "AWARD_DATE": "Certs & awards",
            "LANGUAGE": "Languages", "LANG_LEVEL": "Languages", "SECTION_HEADING": "Heading"}
GROUP_COLORS = {"Name": "#e8590c", "Contact": "#f08c00", "Education": "#d6336c", "Experience": "#1c7ed6", "Skills": "#2f9e44",
                "Projects": "#7048e8", "Certs & awards": "#0c8599", "Languages": "#ae3ec9", "Heading": "#868e96", "Other": "#ced4da"}
REGION_COLORS = {"header": "#1c7ed6", "sidebar": "#15aabf", "column": "#495057", "table": "#12b886", "section_experience": "#1864ab",
                 "section_education": "#d6336c", "section_skills": "#f03e3e", "section_projects": "#94d82d",
                 "section_certifications": "#37b24d", "section_awards": "#ae3ec9", "section_languages": "#22b8cf",
                 "section_contact": "#f08c00", "section_links": "#3bc9db", "section_other": "#adb5bd"}
REGION_TO_GROUP = {"header": "Contact", "section_experience": "Experience", "section_education": "Education", "section_skills": "Skills",
                   "section_projects": "Projects", "section_certifications": "Certs & awards", "section_awards": "Certs & awards",
                   "section_languages": "Languages", "section_contact": "Contact", "section_links": "Contact"}


def group_of(label):
    if not label or label == "O":
        return "Other"
    return GROUP_OF.get(label.split("-", 1)[-1], "Other")


def first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


# =============================================================== dataset access
class Dataset:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.data = self.root / "dataset"
        self.rows = list(csv.DictReader(open(self.data / "manifest.csv", encoding="utf-8"))) if (self.data / "manifest.csv").exists() else []

    def test_resumes(self, layout):
        return sorted({r["resume_id"] for r in self.rows if r["split"] == "test" and r["layout"] == layout})

    def pages(self, rid, tier):
        return sorted((r for r in self.rows if r["resume_id"] == rid and r["tier"] == tier), key=lambda r: int(r["page"]))

    def info(self, rid):
        return next((r for r in self.rows if r["resume_id"] == rid), None)

    def ground_truth(self, rid):
        p = self.data / "ground_truth" / f"{rid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def annotation(self, row):
        return json.loads((self.data / row["annotation"]).read_text(encoding="utf-8"))


# =============================================================== the models
class Models:
    def __init__(self, root=ROOT, log=print):
        self.root = Path(root)
        self.log = log
        import pytesseract
        import spacy
        self.pytesseract = pytesseract
        tess = find_tesseract()
        if not tess:
            raise RuntimeError("Tesseract not found (expected D:\\TesseractOCR\\tesseract.exe)")
        pytesseract.pytesseract.tesseract_cmd = tess
        from pipeline.model_nlp_only import Parser
        self.nlp_parser = Parser(spacy.load("en_core_web_sm", disable=["lemmatizer"]))
        m = self.root / "models"
        vis = self.root / "report_figures" / "vision"
        self.yolo_path = first_existing(m / "yolo_best.pt", vis / "yolo_best.pt")
        self.cnn_path = first_existing(m / "layout_cnn_resnet18.pt", vis / "layout_cnn_resnet18.pt")
        self.llm_paths = {"layout_aware": first_existing(m / "layoutlmv3_layout_aware", m / "best_model_layout_aware"),
                          "text_only": first_existing(m / "layoutlmv3_text_only", m / "best_model_text_only")}
        self._yolo = self._cnn = None
        self._llm = {}

    # ---- lazy loaders
    def yolo(self):
        if self._yolo is None and self.yolo_path:
            from ultralytics import YOLO
            self._yolo = YOLO(str(self.yolo_path))
        return self._yolo

    def cnn(self):
        if self._cnn is None and self.cnn_path:
            import torch
            import torchvision
            net = torchvision.models.resnet18(weights=None)
            net.fc = torch.nn.Linear(net.fc.in_features, 4)
            net.load_state_dict(torch.load(self.cnn_path, map_location="cpu"))
            self._cnn = net.eval()
        return self._cnn

    def llm(self, which):
        if which not in self._llm:
            p = self.llm_paths.get(which)
            if not p:
                self._llm[which] = None
            else:
                from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor
                tok = AutoTokenizer.from_pretrained(p)
                proc = LayoutLMv3ImageProcessor.from_pretrained(p, apply_ocr=False)
                model = LayoutLMv3ForTokenClassification.from_pretrained(p).eval()
                self._llm[which] = (model, tok, proc)
        return self._llm[which]

    def status(self):
        return {"yolo": bool(self.yolo_path), "cnn": bool(self.cnn_path),
                "layout_aware": bool(self.llm_paths["layout_aware"]), "text_only": bool(self.llm_paths["text_only"])}


# =============================================================== stages
def ocr_page(models, img, tier):
    t0 = time.time()
    pp = preprocess(img, tier)
    gray, M = pp["gray"], pp["M"]
    Minv = np.linalg.inv(M)
    t1 = time.time()
    d = models.pytesseract.image_to_data(gray, lang="eng", config=TESS_CONFIG, output_type=models.pytesseract.Output.DICT)
    words = []
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        if not txt or float(d["conf"][i]) < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        poly, bb_orig = map_back(Minv, [x, y, x + w, y + h])
        words.append(dict(id=len(words), text=txt, bbox=[x, y, x + w, y + h], bbox_orig=bb_orig,
                          block=d["block_num"][i], par=d["par_num"][i], line=d["line_num"][i], word=d["word_num"][i]))
    return dict(gray=gray, steps=pp["info"]["steps"], words=words, t_pre=t1 - t0, t_ocr=time.time() - t1)


def raster_lines(words):
    """Same raster flattening as the NLP-only parser, but keeping the word objects (for colouring)."""
    if not words:
        return []
    hs = sorted(w["bbox"][3] - w["bbox"][1] for w in words)
    mh = hs[len(hs) // 2] or 10
    ws = sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2)
    lines, cur, cy = [], [], None
    for w in ws:
        c = (w["bbox"][1] + w["bbox"][3]) / 2
        if cur and abs(c - cy) > 0.5 * mh:
            lines.append(cur)
            cur = []
        cur.append(w)
        cy = c if len(cur) == 1 else (cy * (len(cur) - 1) + c) / len(cur)
    if cur:
        lines.append(cur)
    return [sorted(L, key=lambda w: w["bbox"][0]) for L in lines]


def llm_page(models, which, page):
    """Run one LayoutLMv3 variant on one OCR'd page, with exactly the training-time inputs."""
    import torch
    from PIL import Image
    model, tok, proc = models.llm(which)
    words = page["words"]
    order = raster_order(words)
    texts = [words[i]["text"] for i in order]
    h, w = page["gray"].shape
    boxes = [norm_box(words[i]["bbox"], w, h) for i in order]
    s = 1000 / max(h, w)
    small = cv2.resize(page["gray"], (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(small).convert("RGB")
    if which == "layout_aware":
        labels = predict_page(model, tok, proc, image, texts, boxes, torch.device("cpu"))
    else:  # identical model, no positions and no image
        best = [None] * len(texts)
        for (a, b) in make_windows(tok, texts):
            enc, _ = encode_window(tok, proc, image, texts[a:b], [[0, 0, 0, 0]] * (b - a))
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], bbox=enc["bbox"], pixel_values=None).logits[0]
            pred = logits.argmax(-1).tolist()
            seen = set()
            for ti, wid in enumerate(enc.word_ids(0)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                cent = min(wid, (b - a) - 1 - wid)
                if best[a + wid] is None or cent > best[a + wid][0]:
                    best[a + wid] = (cent, pred[ti])
        labels = [ID2LABEL[x[1]] if x else "O" for x in best]
    return texts, [words[i]["bbox_orig"] for i in order], labels


def cnn_predict(models, img):
    import torch
    from torchvision import transforms as T
    from PIL import Image
    net = models.cnn()
    if net is None:
        return None
    tf = T.Compose([T.Resize((512, 362)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(net(x), 1)[0].tolist()
    return dict(zip(LAYOUTS, p))


def yolo_predict(models, img):
    net = models.yolo()
    if net is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    r = net.predict(small, imgsz=1024, conf=0.25, verbose=False, device="cpu")[0]
    out = []
    for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        out.append(dict(cls=YOLO_NAMES[int(k)], conf=float(c), bbox=[v / s for v in b]))
    return out


# =============================================================== scoring helpers
def resume_f1(gt, pred):
    sc = score_resume(gt, pred)
    tp = sum(v["tp"] for v in sc.values() if v)
    fp = sum(v["fp"] for v in sc.values() if v)
    fn = sum(v["fn"] for v in sc.values() if v)
    return prf(tp, fp, fn)[2], sc


def _entry_text(e, keys):
    return " | ".join(str(e.get(k, "")) for k in keys if e.get(k))


LIST_KEYS = {"education": ("degree", "institution"), "experience": ("title", "company")}


def field_table(gt, preds):
    """Rows of the colour-coded comparison: (field, ground truth, {model: (text, match %)})."""
    rows = []
    for f, label in [("name", "Name"), ("email", "Email"), ("phone", "Phone"), ("location", "Location")]:
        g = SCALARS[f](gt)
        if g:
            rows.append((label, g, {m: ((SCALARS[f](p) or "—"), sim(g, SCALARS[f](p), f) if SCALARS[f](p) else 0.0) for m, p in preds.items()}))
    for key, label in (("experience", "Experience"), ("education", "Education")):
        for i, ge in enumerate((gt.get(key) or [])[:4]):
            gtxt = _entry_text(ge, LIST_KEYS[key])
            cells = {}
            for m, p in preds.items():
                cands = [_entry_text(e, LIST_KEYS[key]) for e in (p.get(key) or [])]
                best = max(cands, key=lambda c: sim(gtxt, c), default="")
                cells[m] = (best or "—", sim(gtxt, best) if best else 0.0)
            rows.append((f"{label} {i + 1}", gtxt, cells))
    for key, label in (("skills", "Skills"), ("certifications", "Certifications"), ("languages", "Languages"), ("projects", "Projects")):
        gl = [LISTS[key](x) for x in (gt.get(key) or [])]
        if not gl:
            continue
        cells = {}
        for m, p in preds.items():
            pl = [LISTS[key](x) for x in (p.get(key) or [])]
            hit = sum(1 for g in gl if any(sim(g, x) >= 90 for x in pl))
            prec = sum(1 for x in pl if any(sim(g, x) >= 90 for g in gl)) / max(1, len(pl))
            rec = hit / len(gl)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            shown = ", ".join(pl[:5]) + (" …" if len(pl) > 5 else "")
            cells[m] = (f"{hit}/{len(gl)} found" + (f": {shown}" if pl else ""), 100 * f1)
        rows.append((label, ", ".join(gl[:5]) + (" …" if len(gl) > 5 else ""), cells))
    return rows


# =============================================================== the full run
def run(models, ds, pages_input, tier, rid=None, progress=lambda stage, frac: None):
    """pages_input: list of dicts with 'img' (BGR) and optionally 'row' (manifest row, dataset resumes).
    Returns a dict with per-page and per-model results."""
    res = dict(rid=rid, tier=tier, pages=[], timings={}, preds={}, source={}, tokens={})
    gt = ds.ground_truth(rid) if rid else None
    res["gt"] = gt
    t = dict(pre=0.0, ocr=0.0, nlp=0.0, text_only=0.0, layout_aware=0.0, cnn=0.0, yolo=0.0)
    for pi, inp in enumerate(pages_input):
        progress(f"Page {pi + 1}: preprocessing + OCR", 0.05 + 0.3 * pi / len(pages_input))
        o = ocr_page(models, inp["img"], tier)
        t["pre"] += o["t_pre"]
        t["ocr"] += o["t_ocr"]
        page = dict(img=inp["img"], gray=o["gray"], steps=o["steps"], words=o["words"], row=inp.get("row"))
        if inp.get("row") is not None:
            ann = ds.annotation(inp["row"])
            page["gt_words"] = ann["words"]
            page["gt_regions"] = ann["regions"]
            proj = project_labels(o["words"], ann["words"])
            for w, (etype, _) in zip(o["words"], proj):
                w["gt_group"] = GROUP_OF.get(etype, "Other") if etype else "Other"
        res["pages"].append(page)

    # NLP-only (raster flattening -> rules + spaCy)
    progress("NLP-only parser", 0.45)
    lines_all = []
    for page in res["pages"]:
        page["raster"] = raster_lines(page["words"])
        lines_all += [" ".join(w["text"] for w in L) for L in page["raster"]]
    t0 = time.time()
    res["preds"]["nlp_only"] = models.nlp_parser.parse(lines_all)
    t["nlp"] = time.time() - t0
    res["source"]["nlp_only"] = "live"

    # LayoutLMv3 (live if weights present, else recorded test-set output)
    for frac, which in ((0.6, "text_only"), (0.75, "layout_aware")):
        progress(MODEL_NICE[which], frac)
        if models.llm_paths.get(which):
            t0 = time.time()
            all_w, all_l, toks = [], [], []
            for page in res["pages"]:
                texts, oboxes, labels = llm_page(models, which, page)
                all_w += texts
                all_l += labels
                toks.append(dict(words=texts, orig_boxes=oboxes, pred=labels))
            res["preds"][which] = decode(all_w, all_l)
            res["tokens"][which] = toks
            t[which] = time.time() - t0
            res["source"][which] = "live"
        elif rid:
            pdir = ROOT / "preds" / which
            pf, tf = pdir / tier / f"{rid}.json", pdir / "_tokens" / tier / f"{rid}.json"
            if pf.exists():
                p = json.loads(pf.read_text(encoding="utf-8"))
                t[which] = p.pop("_runtime_s", 0.0)
                res["preds"][which] = p
                res["tokens"][which] = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else None
                res["source"][which] = "recorded"
            else:
                res["source"][which] = "missing"
        else:
            res["source"][which] = "missing"

    progress("CNN layout classifier", 0.85)
    t0 = time.time()
    res["cnn"] = cnn_predict(models, res["pages"][0]["img"])
    t["cnn"] = time.time() - t0
    progress("YOLO region detector", 0.92)
    t0 = time.time()
    for page in res["pages"]:
        page["yolo"] = yolo_predict(models, page["img"])
    t["yolo"] = time.time() - t0
    res["timings"] = t

    if gt:
        res["f1"], res["field_scores"] = {}, {}
        for m in MODELS:
            if m in res["preds"]:
                res["f1"][m], res["field_scores"][m] = resume_f1(gt, res["preds"][m])
        res["table"] = field_table(gt, {m: res["preds"][m] for m in MODELS if m in res["preds"]})
    progress("Done", 1.0)
    return res
