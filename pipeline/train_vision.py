# %% [markdown]
# # Vision models: YOLO region detector + CNN layout classifier
# **Setup (once):** Add data → your `vision_data` dataset. Settings → Accelerator **GPU T4 x1**, Internet **On**.
# Then **Run All**. Takes about 1–1.5 h. At the end, download `vision_outputs.zip` from the Output panel (right side, `/kaggle/working`).
#
# * Part 1 trains YOLO11-s to find page regions (header, sidebar, columns, tables, and each section type), then measures mAP on the
#   **held-out templates** overall, per class, and per layout × quality tier.
# * Part 2 trains a ResNet-18 CNN to say which of the 4 layouts a page uses, tested on the same held-out templates.
# * If the session stops during YOLO training, just **Run All** again: it resumes from the last epoch.

# %%
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"   # quick local test only; leave as is on Kaggle
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)

hits = glob.glob("/kaggle/input/**/index.csv", recursive=True)
IN = Path(hits[0]).parent if hits else Path(os.environ.get("VISION_DATA", "vision_data"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.environ.get("VISION_WORK", "vision_work"))
DATA = WORK / "vd"
OUT = WORK / "vision_outputs"
OUT.mkdir(parents=True, exist_ok=True)
if not (DATA / "index.csv").exists():
    print("copying data to", DATA)
    shutil.copytree(IN, DATA, dirs_exist_ok=True)
NAMES = (DATA / "classes.txt").read_text().split()
INDEX = list(csv.DictReader(open(DATA / "index.csv", encoding="utf-8")))
for r in INDEX:
    r["path"] = str((DATA / r["file"]).resolve())
LAYOUTS = ["single_column", "multi_column", "table_based", "infographic"]
TIERS = ["clean", "mild", "heavy"]


def write_yaml(path, train, val, test=None, root=None):
    lines = ([f"path: {root}"] if root else []) + [f"train: {train}", f"val: {val}"] + ([f"test: {test}"] if test else [])
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


DATA_YAML = write_yaml(DATA / "data.yaml", "images/train", "images/val", "images/test", root=(DATA / "yolo").resolve())
import torch  # noqa: E402
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("data from", IN, "| pages:", {s: sum(r["split"] == s for r in INDEX) for s in ("train", "val", "test")}, "| GPU:", torch.cuda.is_available())

# %% [markdown]
# ## Part 1 — YOLO region detector

# %%
from ultralytics import YOLO  # noqa: E402

EPOCHS = 2 if SMOKE else 60
IMGSZ = 320 if SMOKE else 1024
BATCH = 4 if SMOKE else 8
RUN = WORK / "runs" / "yolo"
done_flag = RUN / "TRAINING_DONE"
t0 = time.time()
if done_flag.exists():
    print("training already finished, skipping")
elif (RUN / "weights" / "last.pt").exists():
    print("resuming from last.pt")
    YOLO(str(RUN / "weights" / "last.pt")).train(resume=True)
    done_flag.write_text("ok")
else:
    YOLO("yolo11s.pt").train(
        data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, project=str(WORK / "runs"), name="yolo",
        exist_ok=True, patience=20, seed=0, workers=2, plots=True,
        # documents: no flips/mosaic (text is not mirror-symmetric, pages are not collages); small geometric jitter only
        fliplr=0.0, mosaic=0.0, degrees=0.0, translate=0.05, scale=0.25, hsv_h=0.0)
    done_flag.write_text("ok")
print(f"YOLO training time: {(time.time() - t0) / 60:.1f} min")

# %%
best = YOLO(str(RUN / "weights" / "best.pt"))
VAL_KW = dict(imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False)
m = best.val(data=DATA_YAML, split="test", project=str(WORK / "runs"), name="yolo_test", exist_ok=True, plots=True, **VAL_KW)
overall = dict(split="test (held-out templates)", images=sum(r["split"] == "test" for r in INDEX), precision=round(m.box.mp, 4),
               recall=round(m.box.mr, 4), mAP50=round(m.box.map50, 4), mAP50_95=round(m.box.map, 4),
               ms_per_image=round(sum(m.speed.values()), 1))
print(overall)

# instances per class in the test labels
inst = [0] * len(NAMES)
for p in glob.glob(str(DATA / "yolo" / "labels" / "test" / "*.txt")):
    for L in open(p):
        if L.strip():
            inst[int(L.split()[0])] += 1
per_class = []
for i, c in enumerate(m.box.ap_class_index):
    per_class.append(dict(cls=NAMES[int(c)], test_instances=inst[int(c)], precision=round(float(m.box.p[i]), 4),
                          recall=round(float(m.box.r[i]), 4), AP50=round(float(m.box.ap50[i]), 4), AP50_95=round(float(m.box.ap[i]), 4)))
per_class.sort(key=lambda r: -r["test_instances"])

# per layout x tier (and per layout over all tiers)
SUB = DATA / "subsets"
SUB.mkdir(exist_ok=True)
grid = []
for lay in LAYOUTS:
    for tier in TIERS + ["all"]:
        imgs = [r["path"] for r in INDEX if r["split"] == "test" and r["layout"] == lay and (tier == "all" or r["tier"] == tier)]
        if not imgs:
            continue
        lst = SUB / f"{lay}_{tier}.txt"
        lst.write_text("\n".join(imgs) + "\n")
        y = write_yaml(SUB / f"{lay}_{tier}.yaml", str(lst), str(lst))
        s = best.val(data=y, split="val", plots=False, project=str(WORK / "runs"), name="sub", exist_ok=True, **VAL_KW)
        grid.append(dict(layout=lay, tier=tier, images=len(imgs), precision=round(s.box.mp, 4), recall=round(s.box.mr, 4),
                         mAP50=round(s.box.map50, 4), mAP50_95=round(s.box.map, 4)))
        print(grid[-1])


def save_csv(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


save_csv("yolo_test_overall.csv", [overall])
save_csv("yolo_test_per_class.csv", per_class)
save_csv("yolo_test_layout_tier.csv", grid)

# example predictions: one held-out page per layout, clean and heavy
samples = []
for lay in LAYOUTS:
    for tier in ("clean", "heavy"):
        r = next((r for r in INDEX if r["split"] == "test" and r["layout"] == lay and r["tier"] == tier and r["page"] == "1"), None)
        if r:
            samples.append(r["path"])
best.predict(source=samples, imgsz=IMGSZ, device=DEVICE, conf=0.25, save=True, line_width=3, project=str(WORK / "runs"),
             name="yolo_samples", exist_ok=True, verbose=False)
(OUT / "yolo_samples").mkdir(exist_ok=True)
for p in glob.glob(str(WORK / "runs" / "yolo_samples" / "*.jpg")):
    shutil.copy(p, OUT / "yolo_samples")
for f in ["results.png", "results.csv"]:
    if (RUN / f).exists():
        shutil.copy(RUN / f, OUT / f"yolo_train_{f}")
for f in glob.glob(str(WORK / "runs" / "yolo_test" / "*.png")):
    shutil.copy(f, OUT / f"yolo_test_{Path(f).name}")
shutil.copy(RUN / "weights" / "best.pt", OUT / "yolo_best.pt")
print("Part 1 done")

# %% [markdown]
# ## Part 2 — CNN layout classifier (ResNet-18)

# %%
import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

torch.manual_seed(0)
CNN_EPOCHS = 1 if SMOKE else 15
H, W = (256, 182) if SMOKE else (512, 362)          # A4 aspect ratio
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([T.Resize((H, W)), T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.92, 1.05), fill=255),
                      T.ColorJitter(0.3, 0.3, 0.2), T.ToTensor(), norm])
tf_eval = T.Compose([T.Resize((H, W)), T.ToTensor(), norm])


class Pages(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), LAYOUTS.index(r["layout"])


split = {s: [r for r in INDEX if r["split"] == s] for s in ("train", "val", "test")}
NW = 2 if dev.type == "cuda" else 0
dl = {s: DataLoader(Pages(split[s], tf_train if s == "train" else tf_eval), batch_size=16, shuffle=(s == "train"), num_workers=NW)
      for s in split}
try:
    cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
except Exception as e:   # no internet: train from scratch (much weaker - turn Internet on)
    print("could not download ImageNet weights:", e)
    cnn = torchvision.models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, len(LAYOUTS))
cnn = cnn.to(dev)
opt = torch.optim.AdamW(cnn.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS * len(dl["train"]))
scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)


def evaluate(loader):
    cnn.eval()
    preds, tot, n = [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = cnn(x)
            tot += lossf(out.float(), y).item() * len(y)
            n += len(y)
            preds += out.argmax(1).tolist()
    return preds, tot / max(1, n)


cnn_log, best_loss = [], 1e9
t0 = time.time()
for ep in range(CNN_EPOCHS):
    cnn.train()
    tl, n = 0.0, 0
    for x, y in dl["train"]:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            loss = lossf(cnn(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tl, n = tl + loss.item() * len(y), n + len(y)
    vp, vl = evaluate(dl["val"])
    vacc = float(np.mean([p == LAYOUTS.index(r["layout"]) for p, r in zip(vp, split["val"])])) if vp else 0.0
    cnn_log.append(dict(epoch=ep + 1, train_loss=round(tl / n, 4), val_loss=round(vl, 4), val_acc=round(vacc, 4)))
    print(cnn_log[-1])
    if vl <= best_loss:
        best_loss = vl
        torch.save(cnn.state_dict(), OUT / "layout_cnn_resnet18.pt")
print(f"CNN training time: {(time.time() - t0) / 60:.1f} min")
save_csv("cnn_training_log.csv", cnn_log)

# %%
cnn.load_state_dict(torch.load(OUT / "layout_cnn_resnet18.pt", map_location=dev))
t0 = time.time()
tp, _ = evaluate(dl["test"])
ms = 1000 * (time.time() - t0) / max(1, len(tp))
rows = [dict(file=r["file"], resume_id=r["resume_id"], page=r["page"], tier=r["tier"], template=r["template"], true=r["layout"],
             pred=LAYOUTS[p]) for p, r in zip(tp, split["test"])]
save_csv("cnn_test_predictions.csv", rows)
acc = lambda rs: round(float(np.mean([r["true"] == r["pred"] for r in rs])), 4) if rs else ""
summary = [dict(layout=lay, **{t: acc([r for r in rows if r["true"] == lay and r["tier"] == t]) for t in TIERS},
                all=acc([r for r in rows if r["true"] == lay])) for lay in LAYOUTS]
summary.append(dict(layout="ALL", **{t: acc([r for r in rows if r["tier"] == t]) for t in TIERS}, all=acc(rows)))
save_csv("cnn_test_accuracy_layout_tier.csv", summary)
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
yt = [LAYOUTS.index(r["true"]) for r in rows]
yp = [LAYOUTS.index(r["pred"]) for r in rows]
rep = classification_report(yt, yp, labels=range(4), target_names=LAYOUTS, digits=3, zero_division=0)
(OUT / "cnn_test_report.txt").write_text(rep + f"\nInference: {ms:.1f} ms per page ({dev.type})\n")
print(rep)
for s in summary:
    print(s)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
cm = confusion_matrix(yt, yp, labels=range(4))
fig, ax = plt.subplots(figsize=(5.5, 4.6))
ax.imshow(cm, cmap="Blues")
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
short = ["Single", "Multi", "Table", "Infographic"]
ax.set_xticks(range(4), short)
ax.set_yticks(range(4), short)
ax.set_xlabel("Predicted layout")
ax.set_ylabel("True layout")
ax.set_title(f"CNN layout classifier, held-out templates (acc {acc(rows):.2f})", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "cnn_confusion_matrix.png", dpi=200)

# %% [markdown]
# ## Package outputs → download `vision_outputs.zip`

# %%
z = shutil.make_archive(str(WORK / "vision_outputs"), "zip", root_dir=OUT)
print("Download from the Output panel:", z)
for p in sorted(OUT.rglob("*")):
    print(" ", p.relative_to(OUT))
