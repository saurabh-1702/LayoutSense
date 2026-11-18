# Vision deliverables: YOLO region detector + CNN layout classifier

Both run in one Kaggle notebook, `pipeline/train_vision_kaggle.ipynb`, from one uploaded dataset.

## 1. Build the upload (your PC, ~3 min)
```
cd /d D:\projectss\resume_synth
.venv\Scripts\activate
python pipeline\vision_prep.py --data dataset --out vision_data
```
→ `vision_data.zip` (about 250–350 MB): page images resized to 1280 px, YOLO labels, `index.csv` (layout/tier/split per page).

## 2. Kaggle
1. kaggle.com → **Create → New Dataset** → upload `vision_data.zip` → name it `vision-data` → Create.
2. **Create → New Notebook** → File → **Import Notebook** → `train_vision_kaggle.ipynb`.
3. Right panel: **Add Input** → your `vision-data` dataset. Settings: **Accelerator GPU T4 x1**, **Internet On**.
4. **Run All** (about 1–1.5 h). If it stops during YOLO training, Run All again; it resumes.
5. Output panel (`/kaggle/working`) → download **vision_outputs.zip**.

## 3. What you get (unzip into `report_figures\vision`)
| File | Meaning |
|---|---|
| yolo_test_overall.csv | precision, recall, mAP@50, mAP@50-95 on held-out templates; ms per image |
| yolo_test_per_class.csv | AP per region class (header, sidebar, column, table, section_*) |
| yolo_test_layout_tier.csv | mAP per layout × quality tier |
| yolo_train_results.png | training curves |
| yolo_test_*.png | confusion matrix, PR curve |
| yolo_samples/ | predicted boxes on one held-out page per layout (clean + heavy) |
| cnn_test_accuracy_layout_tier.csv | layout-classification accuracy per layout × tier |
| cnn_test_report.txt | precision/recall/F1 per layout + inference time |
| cnn_confusion_matrix.png | which layouts get confused |
| cnn_training_log.csv, cnn_test_predictions.csv | per-epoch log, per-page predictions |
| yolo_best.pt, layout_cnn_resnet18.pt | trained weights |

Test = the 4 held-out templates (never seen in training), same split as the parsers.
