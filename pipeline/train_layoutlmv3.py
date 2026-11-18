# %% [markdown]
# # Model B - Layout-Aware resume parser (LayoutLMv3)
#
# Fine-tunes `microsoft/layoutlmv3-base` for token classification on the resume dataset, then parses
# every **test** resume (held-out templates, all three quality tiers) into the answer-key JSON schema.
#
# **Before running (Kaggle):** Settings -> Accelerator = **GPU T4 x2** (or P100), Internet = **On**,
# and attach your `layoutlm_data` dataset (Add Input). Then *Run All*. Takes about 30-50 minutes.
#
# Inputs are the SAME OCR words, in the SAME raster order, that the NLP-only model reads; LayoutLMv3
# additionally gets each word's box and the page image. So any accuracy difference comes from layout awareness.
#
# Outputs (Output panel -> download): `preds_layout_aware.zip` and `training_log.csv`
# (the fine-tuned model is in `best_model/` if you want it).
#
# **Ablation:** set `TEXT_ONLY = True` in the next cell to train the *same* model on the *same* words in the
# *same* raster order but WITHOUT layout (all boxes = 0, no page image). Outputs then go to
# `preds_text_only.zip`. Comparing the two runs separates "it is a trained model" from "it uses layout".

# %%
import glob, json, math, os, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path

# ---- find the uploaded dataset folder (the one containing examples.jsonl) ----
SMOKE = os.environ.get("LAYOUTLM_SMOKE") == "1"   # local code test with a tiny random model
cands = glob.glob("/kaggle/input/**/examples.jsonl", recursive=True) + glob.glob(os.environ.get("LAYOUTLM_DATA", "layoutlm_data") + "/examples.jsonl")
assert cands, "examples.jsonl not found - attach the layoutlm_data dataset (Add Input) and re-run"
DATA = Path(cands[0]).parent
WORK = Path(os.environ.get("LAYOUTLM_WORK", "/kaggle/working" if Path("/kaggle/working").exists() else "layoutlm_work"))
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DATA))
TEXT_ONLY = os.environ.get("TEXT_ONLY") == "1"   # <- set to True for the no-layout ablation run
import layoutlm_lib
from layoutlm_lib import LABELS, LABEL2ID, ID2LABEL, encode_window, decode


def make_windows(tokenizer, words, max_tokens=500, overlap_words=48):
    """Split a page into <=512-token windows. Token counts use an explicit padding=False call:
    tokenizer.tokenize() inherits padding=max_length after padded encodes, which would make every
    word look 512 tokens long (one-word windows)."""
    if not words:
        return []
    enc = tokenizer(list(words), boxes=[[0, 0, 0, 0]] * len(words), add_special_tokens=False, padding=False, truncation=False)
    lens = [0] * len(words)
    for wid in enc.word_ids():
        if wid is not None:
            lens[wid] += 1
    lens = [max(1, n) for n in lens]
    wins, start = [], 0
    while start < len(words):
        tot, end = 0, start
        while end < len(words) and tot + lens[end] <= max_tokens:
            tot += lens[end]; end += 1
        end = max(end, start + 1)
        wins.append((start, end))
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)
    return wins


layoutlm_lib.make_windows = make_windows
RUN = "text_only" if TEXT_ONLY else "layout_aware"
print("data:", DATA, "| work:", WORK, "| run:", RUN)

# %%
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (LayoutLMv3Config, LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor,
                          LayoutLMv3TokenizerFast, get_linear_schedule_with_warmup)

CFG = dict(
    model_name="microsoft/layoutlmv3-base",
    epochs=12, lr=3e-5, weight_decay=0.01, warmup=0.06,
    batch_size=4, grad_accum=2, max_tokens=500, seed=42,
    train_splits=["train"], val_splits=["val"], predict_splits=["test"],
    patience=4,   # stop if val F1 has not improved for this many epochs
)
if SMOKE:
    CFG.update(epochs=int(os.environ.get("SMOKE_EPOCHS", 3)), lr=1e-3, batch_size=4, grad_accum=1, max_tokens=500, patience=99)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

examples = [json.loads(l) for l in open(DATA / "examples.jsonl", encoding="utf-8")]
by_split = defaultdict(list)
for e in examples:
    by_split[e["split"]].append(e)
print({k: len(v) for k, v in by_split.items()}, "pages")

# %%
# ---- tokenizer, image processor, model ----
if SMOKE:
    tok_dir = Path(os.environ["LAYOUTLM_SMOKE_TOKENIZER"])
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(str(tok_dir))
    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    config = LayoutLMv3Config(vocab_size=tokenizer.vocab_size + 10, hidden_size=96, coordinate_size=16, shape_size=16, num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=192, max_position_embeddings=514, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model = LayoutLMv3ForTokenClassification(config)
else:
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(CFG["model_name"])
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(CFG["model_name"], apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(CFG["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
model.to(device)


class Windows(Dataset):
    """One item = one <=512-token window of one page (long pages are split with overlap)."""
    def __init__(self, exs):
        self.exs, self.items = exs, []
        for ei, e in enumerate(exs):
            for s, t in make_windows(tokenizer, e["words"], CFG["max_tokens"]):
                self.items.append((ei, s, t))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ei, s, t = self.items[i]
        e = self.exs[ei]
        img = Image.open(DATA / e["image"]).convert("RGB")
        boxes = [[0, 0, 0, 0]] * (t - s) if TEXT_ONLY else e["boxes"][s:t]
        enc, pix = encode_window(tokenizer, image_processor, img, e["words"][s:t], boxes, e["labels"][s:t])
        item = dict(input_ids=enc["input_ids"][0], attention_mask=enc["attention_mask"][0], bbox=enc["bbox"][0], labels=enc["labels"][0])
        if not TEXT_ONLY:
            item["pixel_values"] = pix[0]
        return item


train_ds = Windows([e for sp in CFG["train_splits"] for e in by_split[sp]])
val_exs = [e for sp in CFG["val_splits"] for e in by_split[sp]]
val_ds = Windows(val_exs)
train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
val_dl = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=2)
print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")

# %%
def token_f1(model, dl):
    """Micro F1 over non-O tokens (entity type only, B/I ignored) - used to pick the best epoch."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], bbox=b["bbox"], pixel_values=b.get("pixel_values")).logits
            pred, gold = logits.argmax(-1), b["labels"]
            m = gold != -100
            p = [ID2LABEL[i].split("-")[-1] for i in pred[m].tolist()]
            g = [ID2LABEL[i].split("-")[-1] for i in gold[m].tolist()]
            for x, y in zip(p, g):
                if x != "O" and x == y:
                    tp += 1
                else:
                    fp += x != "O"
                    fn += y != "O"
    model.train()
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
steps = math.ceil(len(train_dl) / CFG["grad_accum"]) * CFG["epochs"]
sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

# ---- resume from checkpoint if a previous session timed out ----
CKPT, BEST = WORK / f"checkpoint_{RUN}", WORK / f"best_model_{RUN}"
start_epoch, best_f1, bad, log = 0, -1.0, 0, []
if (CKPT / "state.pt").exists():
    st = torch.load(CKPT / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    start_epoch, best_f1, bad, log = st["epoch"] + 1, st["best_f1"], st["bad"], st["log"]
    print(f"Resumed from epoch {start_epoch}, best val F1 {best_f1:.4f}")

# %%
model.train()
for epoch in range(start_epoch, CFG["epochs"]):
    t0, tot, n = time.time(), 0.0, 0
    for step, b in enumerate(train_dl):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = model(**b).loss / CFG["grad_accum"]
        scaler.scale(loss).backward()
        if (step + 1) % CFG["grad_accum"] == 0 or step + 1 == len(train_dl):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        tot += loss.item() * CFG["grad_accum"]; n += 1
    f1 = token_f1(model, val_dl)
    log.append(dict(epoch=epoch + 1, train_loss=round(tot / max(1, n), 4), val_token_f1=round(f1, 4), minutes=round((time.time() - t0) / 60, 1)))
    print(log[-1], flush=True)
    if f1 > best_f1:
        best_f1, bad = f1, 0
        model.save_pretrained(BEST); tokenizer.save_pretrained(BEST); image_processor.save_pretrained(BEST)
    else:
        bad += 1
    CKPT.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch, best_f1=best_f1, bad=bad, log=log), CKPT / "state.pt")
    if bad >= CFG["patience"]:
        print("early stop"); break
with open(WORK / f"training_log_{RUN}.csv", "w") as f:
    f.write("epoch,train_loss,val_token_f1,minutes\n" + "".join(f"{r['epoch']},{r['train_loss']},{r['val_token_f1']},{r['minutes']}\n" for r in log))
print("best val token F1:", best_f1)

# %% [markdown]
# ## Export the trained model (for the live demo on your PC)
# Download `model_<run>.zip` from the Output panel, then unzip it into `D:\projectss\resume_synth\models\layoutlmv3_<run>`.
# You can stop the notebook after this cell if you only need the model (the next cell re-creates the test predictions).

# %%
z = shutil.make_archive(str(WORK / f"model_{RUN}"), "zip", BEST)
print("Download from the Output panel:", z, f"({Path(z).stat().st_size / 1e6:.0f} MB)")

# %%
# ---- parse every test resume with the best model ----
def predict_page_run(model, img, words, boxes):
    best = [None] * len(words)
    for (s, e) in make_windows(tokenizer, words, CFG["max_tokens"]):
        bx = [[0, 0, 0, 0]] * (e - s) if TEXT_ONLY else boxes[s:e]
        enc, pix = encode_window(tokenizer, image_processor, img, words[s:e], bx)
        wids = enc.word_ids(0)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), bbox=enc["bbox"].to(device),
                           pixel_values=None if TEXT_ONLY else pix.to(device)).logits[0]
        pred = logits.argmax(-1).tolist()
        seen = set()
        for ti, wid in enumerate(wids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            cent = min(wid, (e - s) - 1 - wid)
            if best[s + wid] is None or cent > best[s + wid][0]:
                best[s + wid] = (cent, pred[ti])
    return [ID2LABEL[b[1]] if b else "O" for b in best]


model = LayoutLMv3ForTokenClassification.from_pretrained(BEST).to(device).eval()
OUT = WORK / "preds" / RUN
shutil.rmtree(OUT, ignore_errors=True)
pages = defaultdict(list)
for e in examples:
    if e["split"] in CFG["predict_splits"]:
        pages[(e["resume_id"], e["tier"])].append(e)
start = time.time()
for k, ((rid, tier), exs) in enumerate(sorted(pages.items())):
    exs.sort(key=lambda e: e["page"])
    t0 = time.time()
    all_w, all_l, tok_out = [], [], []
    for e in exs:
        img = Image.open(DATA / e["image"]).convert("RGB")
        labs = predict_page_run(model, img, e["words"], e["boxes"])
        all_w += e["words"]; all_l += labs
        tok_out.append(dict(page=e["page"], words=e["words"], orig_boxes=e["orig_boxes"], pred=labs))
    pred = {"resume_id": rid, **decode(all_w, all_l), "_runtime_s": round(time.time() - t0, 3)}
    (OUT / tier).mkdir(parents=True, exist_ok=True)
    (OUT / tier / f"{rid}.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "_tokens" / tier).mkdir(parents=True, exist_ok=True)
    (OUT / "_tokens" / tier / f"{rid}.json").write_text(json.dumps(tok_out, ensure_ascii=False), encoding="utf-8")
    if (k + 1) % 25 == 0:
        print(f"{k + 1}/{len(pages)} done ({(time.time() - start) / (k + 1):.2f}s per resume)", flush=True)
shutil.make_archive(str(WORK / f"preds_{RUN}"), "zip", WORK / "preds", RUN)
print("Done ->", WORK / f"preds_{RUN}.zip")
