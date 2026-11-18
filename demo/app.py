"""Live demo:  streamlit run demo\\app.py   (from D:\\projectss\\resume_synth, with the venv active)"""
import csv
import html
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from engine import (GROUP_COLORS, GROUP_OF, LAYOUT_NICE, LAYOUTS, MODEL_NICE, MODELS, REGION_COLORS, REGION_TO_GROUP, ROOT,
                    Dataset, Models, group_of, project_labels, run, sim)

st.set_page_config(page_title="Resume Parsing Demo", page_icon="📄", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}
.card{border:1px solid #e3e3e3;border-radius:10px;padding:12px 14px;background:#fafafa;height:100%}
.card h4{margin:0 0 4px 0;font-size:0.95rem;color:#444}
.card .big{font-size:2rem;font-weight:700;line-height:1.1}
.card .small{font-size:0.8rem;color:#666}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;margin-left:6px}
.live{background:#d3f9d8;color:#2b8a3e}.recorded{background:#fff3bf;color:#8a6d00}.missing{background:#ffe3e3;color:#c92a2a}
table.cmp{border-collapse:collapse;width:100%;font-size:0.86rem}
table.cmp th{background:#343a40;color:#fff;padding:6px 8px;text-align:left}
table.cmp td{border:1px solid #dee2e6;padding:5px 8px;vertical-align:top}
.g{background:#c6efce}.y{background:#ffeb9c}.r{background:#ffc7ce}
.flat{font-family:Consolas,monospace;font-size:0.8rem;line-height:1.55;background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:10px;max-height:760px;overflow:auto}
.flat .ln{white-space:nowrap}.flat .mix{background:#fff4e6}
.legend span{display:inline-block;margin:2px 10px 2px 0;font-size:0.8rem}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.rec{font-size:0.85rem;line-height:1.4;border:1px solid #e3e3e3;border-radius:8px;padding:8px 10px;background:#fff}
.rec .sec{font-weight:700;margin:8px 0 2px;border-bottom:1px solid #eee}.rec .it{padding:2px 6px;margin:2px 0;border-radius:4px}
.rec .ind{margin-left:14px}.rec .sm{font-size:0.76rem;color:#555}.rec .meta{font-size:0.76rem;color:#666}.rec .na{color:#adb5bd}
.rec .chips{margin:2px 0}.rec .chip{display:inline-block;border:1px solid #ced4da;border-radius:10px;padding:0 7px;margin:2px;font-size:0.78rem}
.rec .missed{margin-top:8px;padding:6px 8px;background:#fff5f5;border:1px dashed #ffa8a8;border-radius:6px;font-size:0.78rem;color:#a61e1e}
.verdict{border-left:5px solid #1c7ed6;background:#e7f5ff;padding:10px 14px;border-radius:6px;font-size:1.02rem;margin:8px 0 4px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first time only, ~30-60 s)...")
def load():
    return Models(ROOT), Dataset(ROOT)


models, ds = load()
status = models.status()


# ------------------------------------------------------------------ helpers
def color_rgb(hexc, a=None):
    h = hexc.lstrip("#")
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return c + (a,) if a is not None else c


def to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def shrink(im, width=900):
    s = width / im.width
    return im.resize((width, int(im.height * s)), Image.LANCZOS) if s < 1 else im


def font(sz):
    for f in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def paint_boxes(img_bgr, items, alpha=95, outline_wrong=None):
    """items: [(bbox, group)] painted as translucent fills; outline_wrong: [bbox] drawn with a thick red frame."""
    base = to_pil(img_bgr).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for b, g in items:
        if g in ("Other", None):
            continue
        d.rectangle([b[0], b[1], b[2], b[3]], fill=color_rgb(GROUP_COLORS[g], alpha))
    for b in outline_wrong or []:
        d.rectangle([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], outline=(224, 49, 49, 255), width=max(3, base.width // 500))
    return Image.alpha_composite(base, layer).convert("RGB")


def paint_regions(img_bgr, regions, with_conf=True):
    im = to_pil(img_bgr)
    d = ImageDraw.Draw(im)
    lw = max(3, im.width // 350)
    f = font(max(14, im.width // 55))
    for r in regions:
        b, c = r["bbox"], REGION_COLORS.get(r["cls"], "#555555")
        d.rectangle(b, outline=color_rgb(c), width=lw)
        label = r["cls"].replace("section_", "") + (f" {r['conf']:.2f}" if with_conf and "conf" in r else "")
        tb = d.textbbox((b[0], b[1]), label, font=f)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color_rgb(c))
        d.text((b[0], b[1]), label, fill=(255, 255, 255), font=f)
    return im


def legend(groups):
    return "<div class='legend'>" + "".join(f"<span><i style='background:{GROUP_COLORS[g]}'></i>{g}</span>" for g in groups) + "</div>"


def badge(src):
    return {"live": "<span class='badge live'>LIVE</span>", "recorded": "<span class='badge recorded'>KAGGLE TEST-RUN OUTPUT</span>",
            "missing": "<span class='badge missing'>NOT AVAILABLE</span>"}.get(src, "")


def cls_of(score):
    return "g" if score >= 90 else ("y" if score >= 50 else "r")


def word_group_live(page):
    """Section of every OCR word: from the answer key if we have one, else from the YOLO region containing it."""
    if page.get("gt_words") is not None:
        return {w["id"]: w.get("gt_group", "Other") for w in page["words"]}
    out = {}
    for w in page["words"]:
        cx, cy = (w["bbox_orig"][0] + w["bbox_orig"][2]) / 2, (w["bbox_orig"][1] + w["bbox_orig"][3]) / 2
        g = "Other"
        for r in sorted(page.get("yolo") or [], key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])):
            b = r["bbox"]
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and r["cls"] in REGION_TO_GROUP:
                g = REGION_TO_GROUP[r["cls"]]
                break
        out[w["id"]] = g
    return out


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📄 Resume Parsing Demo")
st.sidebar.caption("NLP-only vs layout-aware parsing, live on one resume")
mode = st.sidebar.radio("Input", ["Test resume (unseen design, with answer key)", "Upload a resume image"], index=0)
pages_input, tier, rid = None, "clean", None
if mode.startswith("Test"):
    lay = st.sidebar.selectbox("Layout type", LAYOUTS, format_func=lambda l: LAYOUT_NICE[l], index=1)
    tier = st.sidebar.radio("Image quality", ["clean", "mild", "heavy"], horizontal=True)
    ids = ds.test_resumes(lay)
    if not ids:
        st.sidebar.error("No test resumes found - run from the project folder (dataset\\manifest.csv).")
    else:
        def describe(r):
            i = ds.info(r)
            return f"{r} · {i['template']} · {i['level']} · {i['industry']}" if i else r
        rid = st.sidebar.selectbox("Resume", ids, format_func=describe)
        rows = ds.pages(rid, tier)
        pages_input = [dict(img=cv2.imread(str(ds.data / r["image"])), row=r) for r in rows]
        st.sidebar.image(to_pil(pages_input[0]["img"]), caption=f"{rid} page 1 of {len(rows)} ({tier})", width="stretch")
else:
    up = st.sidebar.file_uploader("Resume page image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    tier = st.sidebar.radio("Image quality (controls preprocessing)", ["clean", "mild", "heavy"], horizontal=True,
                            help="clean = digital image; mild/heavy = scanned or photographed (turns on page detection, deskew, denoise, CLAHE)")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        pages_input = [dict(img=img)]
        st.sidebar.image(to_pil(img), width="stretch")
    st.sidebar.info("Uploaded images have no answer key, so outputs are shown but not scored. "
                    "LayoutLMv3 outputs exist only for the dataset's test resumes (they were computed on Kaggle), "
                    "so an upload runs preprocessing, OCR, the NLP-only parser, the CNN and YOLO.")

go = st.sidebar.button("▶  Run all models", type="primary", width="stretch", disabled=not pages_input)
st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
for k, nm in (("layout_aware", "LayoutLMv3 layout-aware"), ("text_only", "LayoutLMv3 text-only"), ("cnn", "CNN layout classifier"), ("yolo", "YOLO11 region detector")):
    st.sidebar.markdown(f"{'🟢' if status[k] else '🟡'} {nm} — {'live' if status[k] else ('outputs from the Kaggle test run' if k in ('layout_aware', 'text_only') else 'weights missing')}",
                        help=None)
st.sidebar.markdown("🟢 Preprocessing, Tesseract OCR, NLP-only parser — live")

# ------------------------------------------------------------------ run
st.title("Does layout awareness matter for resume parsing?")
st.caption("One resume → shared CV preprocessing & OCR → three parsers (rules / LayoutLMv3 text-only / LayoutLMv3 layout-aware) "
           "+ CNN layout classifier + YOLO region detector. Test resumes come from designs the models never saw in training.")

if go:
    bar = st.progress(0.0, text="Starting…")
    t0 = time.time()
    res = run(models, ds, pages_input, tier, rid, progress=lambda s, f: bar.progress(min(1.0, f), text=s))
    res["wall"] = time.time() - t0
    st.session_state["res"] = res
    bar.empty()

res = st.session_state.get("res")
if not res:
    st.info("Pick a resume in the sidebar and press **Run all models**.  \n"
            "Suggested demo: run one **Multi-column** resume (clean), then the same layout with **heavy** quality, then a **Single-column** one.")
    st.stop()

gt = res["gt"]
lay_true = ds.info(res["rid"])["layout"] if res["rid"] else None

# ------------------------------------------------------------------ headline cards
cols = st.columns(5)
for c, m in zip(cols[:3], MODELS):
    src = res["source"].get(m, "missing")
    f1 = res.get("f1", {}).get(m)
    tm = res["timings"].get(m if m != "nlp_only" else "nlp", 0)
    col = "#2b8a3e" if (f1 or 0) >= 0.9 else ("#e67700" if (f1 or 0) >= 0.5 else "#c92a2a")
    big = f"<div class='big' style='color:{col}'>{f1:.2f}</div><div class='small'>resume F1 vs answer key</div>" if f1 is not None else \
          ("<div class='big'>—</div><div class='small'>no answer key (uploaded image)</div>" if src != "missing" else "<div class='big'>✖</div><div class='small'>not available for this input</div>")
    c.markdown(f"<div class='card'><h4>{MODEL_NICE[m]}{badge(src)}</h4>{big}<div class='small'>parse time {tm:.2f} s{' (Kaggle T4 GPU)' if src == 'recorded' else ''}</div></div>", unsafe_allow_html=True)
if res.get("cnn"):
    pred = max(res["cnn"], key=res["cnn"].get)
    ok = "" if not lay_true else ("✅ correct" if pred == lay_true else f"❌ true: {LAYOUT_NICE[lay_true]}")
    cols[3].markdown(f"<div class='card'><h4>CNN layout classifier<span class='badge live'>LIVE</span></h4><div class='big' style='font-size:1.35rem'>{LAYOUT_NICE[pred]}</div>"
                     f"<div class='small'>confidence {res['cnn'][pred]:.0%} · {ok}</div><div class='small'>{res['timings']['cnn']:.2f} s</div></div>", unsafe_allow_html=True)
nreg = sum(len(p.get("yolo") or []) for p in res["pages"])
cols[4].markdown(f"<div class='card'><h4>YOLO region detector<span class='badge live'>LIVE</span></h4><div class='big'>{nreg}</div>"
                 f"<div class='small'>regions found on {len(res['pages'])} page(s)</div><div class='small'>{res['timings']['yolo']:.2f} s</div></div>", unsafe_allow_html=True)

t = res["timings"]
st.caption(f"Shared stages: preprocessing {t['pre']:.1f} s · Tesseract OCR {t['ocr']:.1f} s · {sum(len(p['words']) for p in res['pages'])} words read · "
           f"steps: {' → '.join(res['pages'][0]['steps'])} · total wall time {res['wall']:.1f} s")
if res.get("f1"):
    f = res["f1"]
    best = max(f, key=f.get)
    lay_txt = LAYOUT_NICE.get(lay_true, "")
    msg = f"On this <b>{lay_txt.lower()}</b> resume ({res['tier']} image) the <b>{MODEL_NICE[best]}</b> scores best: " + \
          ", ".join(f"{MODEL_NICE[m].split(' (')[0]} <b>{f[m]:.2f}</b>" for m in MODELS if m in f) + "."
    if "layout_aware" in f and "text_only" in f:
        d = f["layout_aware"] - f["text_only"]
        msg += f" Layout information (layout-aware − text-only) adds <b>{d:+.2f}</b>."
    st.markdown(f"<div class='verdict'>{msg}</div>", unsafe_allow_html=True)

tabs = st.tabs(["1 · The resume & regions", "2 · What the NLP model read", "3 · What LayoutLMv3 read",
                "4 · Parsed output of every model", "5 · Field-by-field scores", "6 · Where each model went wrong", "7 · Whole test set"])

# ------------------------------------------------------------------ tab 1
with tabs[0]:
    pi = st.radio("Page", list(range(len(res["pages"]))), format_func=lambda i: f"Page {i + 1}", horizontal=True, key="p1") if len(res["pages"]) > 1 else 0
    page = res["pages"][pi]
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Input image**")
    c1.image(shrink(to_pil(page["img"])), width="stretch")
    c2.markdown("**After CV preprocessing** (what OCR reads)")
    c2.image(shrink(Image.fromarray(page["gray"])), width="stretch")
    c2.caption(" → ".join(page["steps"]))
    c3.markdown("**YOLO region detection**")
    if page.get("yolo") is not None:
        c3.image(shrink(paint_regions(page["img"], page["yolo"])), width="stretch")
    else:
        c3.warning("YOLO weights not found (models\\yolo_best.pt).")
    if res.get("cnn"):
        st.markdown("**CNN layout classifier — probabilities**")
        st.bar_chart({LAYOUT_NICE[k]: v for k, v in res["cnn"].items()}, horizontal=True, height=180)

# ------------------------------------------------------------------ tab 2
with tabs[1]:
    st.markdown("The NLP-only parser reads OCR lines **straight across the whole page width** (raster order). "
                "Each word below is coloured by the section it really belongs to"
                + (" (from the answer key)." if gt else " (from the YOLO regions).")
                + " **Lines that mix colours are lines where two columns were glued together** — the NLP model cannot undo this.")
    pi = st.radio("Page", list(range(len(res["pages"]))), format_func=lambda i: f"Page {i + 1}", horizontal=True, key="p2") if len(res["pages"]) > 1 else 0
    page = res["pages"][pi]
    groups = word_group_live(page)
    used = [g for g in GROUP_COLORS if g not in ("Other",) and g in set(groups.values())]
    st.markdown(legend(used + ["Other"]), unsafe_allow_html=True)
    mixed, out = 0, []
    for L in page["raster"]:
        gs = {groups.get(w["id"], "Other") for w in L} - {"Other", "Heading"}
        mix = len(gs) >= 2
        mixed += mix
        spans = " ".join(f"<span style='color:{GROUP_COLORS[groups.get(w['id'], 'Other')] if groups.get(w['id'], 'Other') != 'Other' else '#999'};"
                         f"font-weight:{600 if groups.get(w['id'], 'Other') != 'Other' else 400}'>{html.escape(w['text'])}</span>" for w in L)
        out.append(f"<div class='ln{' mix' if mix else ''}'>{'⚠️ ' if mix else '&nbsp;&nbsp;&nbsp;'}{spans}</div>")
    c1, c2 = st.columns([1, 1.15])
    c1.image(shrink(paint_boxes(page["img"], [(w["bbox_orig"], groups.get(w["id"], "Other")) for w in page["words"]], alpha=80)), width="stretch")
    c2.metric("Flattened lines that mix two or more sections", f"{mixed} of {len(page['raster'])}")
    c2.markdown(f"<div class='flat'>{''.join(out)}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 3 (LayoutLMv3)
def token_group_for_words(page, tk):
    """Give every OCR word the field LayoutLMv3 predicted for the token at the same place on the page."""
    cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in tk["orig_boxes"]]
    out = {}
    for w in page["words"]:
        b = w["bbox_orig"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best, bd = None, 1e18
        for j, (x, y) in enumerate(cents):
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < bd:
                best, bd = j, d
        h = max(4.0, b[3] - b[1])
        out[w["id"]] = group_of(tk["pred"][best]) if best is not None and bd < (1.5 * h) ** 2 else "Other"
    return out


def _match_cls(score):
    return "" if score is None else ("g" if score >= 90 else ("y" if score >= 50 else "r"))


def _best(text, cands):
    return max((sim(text, c) for c in cands), default=0.0)


def render_full(p, gt=None, is_key=False):
    """A complete parsed resume as readable HTML. With an answer key, every item is shaded:
    green = matches the key (>=90%), yellow = partly right, red = wrong / not in the key; missed key items are listed at the end."""
    if not p:
        return "<div class='rec'><i>no output for this input</i></div>"
    score = (lambda a, b, f=None: None) if (gt is None or is_key) else (lambda a, b, f=None: sim(a, b, f) if a and b else 0.0)
    c, gc = p.get("contact") or {}, ((gt or {}).get("contact") or {})
    h, missed = [], []

    def head(lbl, g, n=None):
        h.append(f"<div class='sec' style='color:{GROUP_COLORS[g]}'>{lbl}{f' ({n})' if n is not None else ''}</div>")

    def item(text, sc, indent=False, small=False):
        cl = _match_cls(sc)
        h.append(f"<div class='it {cl}{' ind' if indent else ''}{' sm' if small else ''}'>{text}</div>")

    head("Name & contact", "Name")
    for lbl, val, gval, fld in (("Name", p.get("name"), (gt or {}).get("name"), "name"), ("Email", c.get("email"), gc.get("email"), "email"),
                                ("Phone", c.get("phone"), gc.get("phone"), "phone"), ("Location", c.get("location"), gc.get("location"), "location")):
        if val:
            item(f"<b>{lbl}:</b> {html.escape(str(val))}", score(gval, val, fld) if gval else (None if gt is None or is_key else 0.0))
        elif gval and not is_key:
            missed.append(f"{lbl}: {gval}")
        elif is_key or gt is None:
            item(f"<b>{lbl}:</b> <span class='na'>—</span>", None)
    links = [l.get("url") if isinstance(l, dict) else l for l in (p.get("links") or [])]
    glinks = [l.get("url") if isinstance(l, dict) else l for l in ((gt or {}).get("links") or [])]
    for u in links:
        item(f"<b>Link:</b> {html.escape(str(u))}", None if (gt is None or is_key) else _best(u, glinks))

    for key, lbl, g, k1, k2 in (("experience", "Experience", "Experience", "title", "company"), ("education", "Education", "Education", "degree", "institution")):
        items = p.get(key) or []
        gitems = (gt or {}).get(key) or []
        gkeys = [f"{e.get(k1, '')} | {e.get(k2, '')}" for e in gitems]
        head(lbl, g, len(items))
        if not items:
            item("<span class='na'>— none found —</span>", None)
        for e in items:
            sc = None if (gt is None or is_key) else _best(f"{e.get(k1, '')} | {e.get(k2, '')}", gkeys)
            meta = " · ".join(html.escape(str(e[x])) for x in ("dates", "location", "grade") if e.get(x))
            item(f"<b>{html.escape(str(e.get(k1) or '?'))}</b> — {html.escape(str(e.get(k2) or '?'))}" + (f"<br><span class='meta'>{meta}</span>" if meta else ""), sc)
            for b in (e.get("bullets") or [])[:3]:
                item("• " + html.escape(str(b))[:140], None, indent=True, small=True)
            if len(e.get("bullets") or []) > 3:
                item(f"… +{len(e['bullets']) - 3} more", None, indent=True, small=True)
        if gt is not None and not is_key:
            got = [f"{e.get(k1, '')} | {e.get(k2, '')}" for e in items]
            missed += [f"{lbl}: {k.replace(' | ', ' — ')}" for k in gkeys if _best(k, got) < 90]

    for key, lbl, g, get in (("skills", "Skills", "Skills", lambda x: x if isinstance(x, str) else str(x)),
                             ("projects", "Projects", "Projects", lambda x: x.get("name") if isinstance(x, dict) else x),
                             ("certifications", "Certifications", "Certs & awards", lambda x: x.get("name") if isinstance(x, dict) else x),
                             ("awards", "Awards", "Certs & awards", lambda x: x.get("name") if isinstance(x, dict) else x),
                             ("languages", "Languages", "Languages", lambda x: (x.get("language", "") + (f" ({x['proficiency']})" if x.get("proficiency") else "")) if isinstance(x, dict) else x)):
        vals = [get(x) for x in (p.get(key) or []) if get(x)]
        gvals = [get(x) for x in ((gt or {}).get(key) or []) if get(x)]
        if not vals and not gvals:
            continue
        head(lbl, g, len(vals))
        chips = []
        for v in vals:
            cl = "" if (gt is None or is_key) else _match_cls(_best(v.split(" (")[0] if key == "languages" else v,
                                                                     [x.split(" (")[0] for x in gvals] if key == "languages" else gvals))
            chips.append(f"<span class='chip {cl}'>{html.escape(str(v))}</span>")
        h.append("<div class='chips'>" + ("".join(chips) or "<span class='na'>— none found —</span>") + "</div>")
        if gt is not None and not is_key:
            missed += [f"{lbl}: {x}" for x in gvals if _best(x.split(' (')[0] if key == 'languages' else x,
                                                               [v.split(' (')[0] for v in vals] if key == 'languages' else vals) < 90]
    if missed:
        h.append(f"<div class='missed'><b>Missed ({len(missed)}):</b><br>" + "<br>".join(html.escape(m)[:120] for m in missed[:25])
                 + (f"<br>… +{len(missed) - 25} more" if len(missed) > 25 else "") + "</div>")
    return "<div class='rec'>" + "".join(h) + "</div>"


with tabs[2]:
    llms = [m for m in ("layout_aware", "text_only") if res["tokens"].get(m)]
    if not llms:
        st.info("LayoutLMv3 outputs are available for the dataset's test resumes (computed on Kaggle). "
                "Pick a test resume in the sidebar to see them.")
    else:
        st.markdown("LayoutLMv3 receives **exactly the same words in the same jumbled order** as the NLP model — "
                    "but the layout-aware version also gets **each word's position on the page and the page image**. "
                    "Below, the same flattened lines as in tab 2, with every word coloured by the field **LayoutLMv3 predicted**. "
                    + ("<span style='color:#c92a2a;text-decoration:line-through;font-weight:600'>Struck-through red</span> = the model chose the wrong field." if gt else ""),
                    unsafe_allow_html=True)
        which = st.radio("Model", llms, format_func=lambda m: MODEL_NICE[m], horizontal=True, key="llm_pick")
        pi = st.radio("Page", list(range(len(res["pages"]))), format_func=lambda i: f"Page {i + 1}", horizontal=True, key="p3") if len(res["pages"]) > 1 else 0
        page = res["pages"][pi]
        toks = res["tokens"][which]
        if pi < len(toks):
            pred_g = token_group_for_words(page, toks[pi])
            true_g = word_group_live(page) if page.get("gt_words") is not None else {}
            st.markdown(legend([g for g in GROUP_COLORS if g != "Other"]), unsafe_allow_html=True)
            lines_html, n_wrong, n_lab = [], 0, 0
            for L in page["raster"]:
                spans = []
                for w in L:
                    pg_, tg = pred_g.get(w["id"], "Other"), true_g.get(w["id"])
                    wrong = tg is not None and pg_ != tg and not (pg_ in ("Other", "Heading") and tg in ("Other", "Heading"))
                    n_wrong += wrong
                    n_lab += tg is not None and tg not in ("Other",)
                    col = GROUP_COLORS[pg_] if pg_ != "Other" else "#999"
                    style = f"color:{col};font-weight:{600 if pg_ != 'Other' else 400}"
                    if wrong:
                        style = "color:#c92a2a;font-weight:600;text-decoration:line-through"
                    spans.append(f"<span style='{style}' title='predicted: {pg_}{(' | true: ' + tg) if tg else ''}'>{html.escape(w['text'])}</span>")
                lines_html.append(f"<div class='ln'>{' '.join(spans)}</div>")
            c1, c2 = st.columns([1, 1.15])
            c1.image(shrink(paint_boxes(page["img"], [(w["bbox_orig"], pred_g.get(w["id"], "Other")) for w in page["words"]], alpha=80)), width="stretch")
            if gt:
                c2.metric("Words given the wrong field", f"{n_wrong} of {len(page['words'])}")
            c2.markdown(f"<div class='flat'>{''.join(lines_html)}</div>", unsafe_allow_html=True)
            st.caption("Hover a word to see the predicted field (and the true field, for test resumes). "
                       "Compare with tab 2: the NLP model saw these same lines but had no positions to separate the columns.")

# ------------------------------------------------------------------ tab 4 (parsed output of every model)
with tabs[3]:
    avail = [m for m in MODELS if m in res["preds"]]
    if gt:
        st.markdown("The complete resume **as each model parsed it**, next to the answer key. "
                    "<span class='g' style='padding:1px 6px'>green = matches the answer key</span> "
                    "<span class='y' style='padding:1px 6px'>yellow = partly right</span> "
                    "<span class='r' style='padding:1px 6px'>red = wrong / invented</span> · "
                    "anything the model did not find is listed under <b>Missed</b>.", unsafe_allow_html=True)
    else:
        st.markdown("The complete resume **as each model parsed it** (no answer key for an uploaded image, so nothing is coloured).")
    cs = st.columns((1 if gt else 0) + len(avail))
    k = 0
    if gt:
        cs[0].markdown("**Answer key** (ground truth)")
        cs[0].markdown(render_full(gt, gt, is_key=True), unsafe_allow_html=True)
        k = 1
    for c, m in zip(cs[k:], avail):
        f1 = res.get("f1", {}).get(m)
        c.markdown(f"**{MODEL_NICE[m]}**{badge(res['source'].get(m))}" + (f"<br>resume F1 <b>{f1:.2f}</b>" if f1 is not None else ""), unsafe_allow_html=True)
        c.markdown(render_full(res["preds"][m], gt), unsafe_allow_html=True)
    for m in MODELS:
        if m not in res["preds"]:
            st.caption(f"{MODEL_NICE[m]}: not available for this input (LayoutLMv3 outputs exist for the dataset's test resumes).")
    with st.expander("Raw JSON output of each model"):
        cs = st.columns(len(avail) or 1)
        for c, m in zip(cs, avail):
            c.markdown(f"**{MODEL_NICE[m]}**")
            c.json(res["preds"][m], expanded=False)

# ------------------------------------------------------------------ tab 5 (field table)
with tabs[4]:
    avail = [m for m in MODELS if m in res["preds"]]
    if gt and res.get("table"):
        st.markdown("Every cell is compared with the answer key using fuzzy matching (RapidFuzz). "
                    "<span class='g' style='padding:1px 6px'>green ≥ 90% match</span> "
                    "<span class='y' style='padding:1px 6px'>yellow 50–89%</span> "
                    "<span class='r' style='padding:1px 6px'>red &lt; 50% or missing</span>", unsafe_allow_html=True)
        h = "<table class='cmp'><tr><th>Field</th><th>Answer key</th>" + "".join(f"<th>{MODEL_NICE[m]}</th>" for m in avail) + "</tr>"
        for label, g, cells in res["table"]:
            h += f"<tr><td><b>{html.escape(label)}</b></td><td>{html.escape(str(g))}</td>"
            for m in avail:
                txt, sc = cells.get(m, ("—", 0))
                h += f"<td class='{cls_of(sc)}'>{html.escape(str(txt))}<br><span style='font-size:0.72rem;color:#555'>{sc:.0f}% match</span></td>"
            h += "</tr>"
        st.markdown(h + "</table>", unsafe_allow_html=True)
        st.markdown("**Per-field F1 for this resume**")
        fs = res["field_scores"]
        rows = []
        for fld in ["name", "email", "phone", "location", "education", "experience", "skills", "certifications", "languages", "projects"]:
            r = {"field": fld}
            for m in avail:
                v = fs[m].get(fld)
                if v:
                    p = v["tp"] / max(1, v["tp"] + v["fp"])
                    rc = v["tp"] / max(1, v["tp"] + v["fn"])
                    r[MODEL_NICE[m]] = round(2 * p * rc / max(1e-9, p + rc), 2)
            if len(r) > 1:
                rows.append(r)
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No answer key for an uploaded image — showing each model's raw output.")
        cs = st.columns(len(avail) or 1)
        for c, m in zip(cs, avail):
            c.markdown(f"**{MODEL_NICE[m]}**")
            c.json(res["preds"][m], expanded=2)

# ------------------------------------------------------------------ tab 6
with tabs[5]:
    pi = st.radio("Page", list(range(len(res["pages"]))), format_func=lambda i: f"Page {i + 1}", horizontal=True, key="p4") if len(res["pages"]) > 1 else 0
    page = res["pages"][pi]
    st.markdown("Each word is painted with the field the model assigned to it. "
                + ("**Red frames = the model gave this word the wrong field** (compared with the answer key)." if gt else ""))
    st.markdown(legend([g for g in GROUP_COLORS if g != "Other"]), unsafe_allow_html=True)
    panels = []
    if page.get("gt_words") is not None:
        panels.append(("Answer key", paint_boxes(page["img"], [(w["bbox"], group_of(w["label"])) for w in page["gt_words"]])))
    for m in ("text_only", "layout_aware"):
        toks = res["tokens"].get(m)
        if not toks or pi >= len(toks):
            continue
        tk = toks[pi]
        items = [(b, group_of(l)) for b, l in zip(tk["orig_boxes"], tk["pred"])]
        wrong = []
        if page.get("gt_words") is not None:
            proj = project_labels([dict(bbox_orig=b) for b in tk["orig_boxes"]], page["gt_words"])
            for (b, l), (etype, _) in zip(zip(tk["orig_boxes"], tk["pred"]), proj):
                if group_of(l) != (GROUP_OF.get(etype, "Other") if etype else "Other"):
                    wrong.append(b)
        title = f"{MODEL_NICE[m]}" + (f" — {len(wrong)} of {len(items)} words wrong" if page.get('gt_words') is not None else "")
        panels.append((title, paint_boxes(page["img"], items, outline_wrong=wrong)))
    if panels:
        cs = st.columns(len(panels))
        for c, (ttl, im) in zip(cs, panels):
            c.markdown(f"**{ttl}**")
            c.image(shrink(im), width="stretch")
    if gt and res.get("table"):
        st.markdown("#### What each model got wrong")
        cs = st.columns(len([m for m in MODELS if m in res["preds"]]))
        for c, m in zip(cs, [m for m in MODELS if m in res["preds"]]):
            c.markdown(f"**{MODEL_NICE[m]}** — F1 {res['f1'][m]:.2f}")
            errs = [(lab, g, cells[m]) for lab, g, cells in res["table"] if m in cells and cells[m][1] < 90]
            if not errs:
                c.success("Every field matched the answer key.")
            for lab, g, (txt, sc) in errs:
                kind = "missing" if txt in ("—", "") or sc == 0 else "wrong"
                c.markdown(f"<div class='{cls_of(sc)}' style='padding:4px 8px;margin:3px 0;border-radius:5px;font-size:0.84rem'>"
                           f"<b>{html.escape(lab)}</b> ({kind}, {sc:.0f}%)<br>expected: {html.escape(str(g))[:110]}<br>got: {html.escape(str(txt))[:110]}</div>",
                           unsafe_allow_html=True)
        st.caption("The NLP-only parser has no word-level labels to paint — its errors come from the flattened text shown in tab 2.")

# ------------------------------------------------------------------ tab 7
with tabs[6]:
    st.markdown("Results of the full experiment: **100 test resumes from 4 unseen designs × 3 image qualities** (mean per-resume F1).")
    t1 = ROOT / "report_figures" / "figures" / "T1_main_results.csv"
    if t1.exists():
        rows = list(csv.DictReader(open(t1, encoding="utf-8")))
        st.dataframe([{"Layout": LAYOUT_NICE.get(r["layout"], r["layout"]), "Quality": r["tier"], "NLP-only": float(r["nlp_only"]),
                       "Text-only": float(r.get("text_only") or 0), "Layout-aware": float(r["layout_aware"]),
                       "Layout gain (LA − text)": round(float(r["layout_aware"]) - float(r.get("text_only") or 0), 3)} for r in rows],
                     width="stretch", hide_index=True)
    hm = ROOT / "report_figures" / "figures" / "F1_heatmaps.png"
    if hm.exists():
        st.image(str(hm), width="stretch")
    st.markdown("**Key findings:** layout-aware is best in all 12 layout × quality cells (mean F1 0.708 vs 0.406). "
                "Layout itself adds a significant +0.11 to +0.18 on multi-column and +0.08 to +0.12 on clean/mild table resumes, "
                "and nothing significant on single-column or infographic ones. Under heavy degradation OCR becomes the bottleneck for every model, "
                "while the CNN (90.2 % accuracy) and YOLO (0.731 mAP@50) are almost unaffected.")
