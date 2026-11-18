# Does Layout Awareness Matter for Resume Parsing?
### Computer Vision Mini Project: Submission Package

**Team:** Saurabh Pandey (70122300020, I053) · Eshan Mohammed (70122300024, I046) · Siddhant Nikam (70122300016, I051)
B.Tech Artificial Intelligence

This project compares a **rule-based NLP parser** that flattens the page into plain text with **LayoutLMv3**, which reads both the words and their positions on the page. They are tested on 4 resume layouts (single-column, multi-column, table, infographic) and 3 image qualities (clean, mild, heavy). Two extra models are included: a **ResNet-18** that classifies the layout type and a **YOLO11** model that detects resume regions. The research paper (`paper/`) reports all results.

---

## 1. Folder guide

| Folder / file | What it contains |
|---|---|
| `docs/SUBMISSION_GUIDE.md` | This file |
| `docs/DATASET.md` | Detailed description of the synthetic dataset and its answer keys |
| `requirements.txt` | Python packages for the dataset generator and pipelines |
| `generate.py` + `resumegen/` | **Dataset generator.** Jinja2 HTML templates (12), rendered at 300 DPI with Playwright Chromium, plus ground truth read from the page and image degradation (`degrade.py`) |
| `pipeline/preprocess.py` | **Shared CV preprocessing:** grayscale, Otsu, morphology, contours, page detection, perspective rectification, Hough deskew, denoising, illumination flattening, CLAHE |
| `pipeline/run_ocr.py`, `eval_ocr.py` | Tesseract 5 OCR (shared by all parsers) and OCR accuracy evaluation |
| `pipeline/model_nlp_only.py` | **Model A, NLP-only parser:** raster flattening, regex, heading lexicon, spaCy, rules |
| `pipeline/layoutlm_prep.py`, `layoutlm_lib.py`, `train_layoutlmv3.py` | **Model B, LayoutLMv3:** label projection, windowing, training and decoding |
| `pipeline/train_layoutlmv3_kaggle.ipynb` | Kaggle notebook used to train and test **layout-aware** LayoutLMv3 (T4 GPU) |
| `pipeline/ablation_text_only_kaggle.ipynb` | Kaggle notebook for the **text-only ablation** (same model, positions removed) |
| `pipeline/vision_prep.py`, `train_vision.py`, `train_vision_kaggle.ipynb` | **ResNet-18 layout classifier** and **YOLO11 region detector** |
| `score.py` | Field-level scoring: fuzzy match ≥ 90, per-resume micro-F1 |
| `make_figures.py` | Builds all result tables, heatmaps and case figures from `preds/` |
| `tools/` | Project audit, YOLO label check, visualisation, spot checks |
| `demo/` | **Live Streamlit demo:** one resume, every model, side by side (see `docs/DEMO.md`) |
| `preds/` | Saved predictions on the test set: `nlp_only`, `text_only`, `layout_aware`, `label_ceiling` |
| `ocr/` | OCR accuracy results (Table III of the paper) |
| `report_figures/figures/` | All result tables (CSV/MD) and figures |
| `report_figures/vision/` | Trained weights `yolo_best.pt`, `layout_cnn_resnet18.pt`, and their reports |
| `report_figures/training_log*.csv` | LayoutLMv3 validation F1 per epoch (training curves) |
| `dataset/` | **Sample** of the test set: 3 resumes per layout × 3 qualities (the full set is about 1.5 GB and can be regenerated) |

---

## 2. How to run

### Setup (Windows, Python 3.10+)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r demo\requirements_demo.txt
playwright install chromium
python -m spacy download en_core_web_sm
```
Also install **Tesseract OCR 5** (the code looks in `D:\TesseractOCR\` and on the PATH).

### A. Run the live demo (quickest way to see the project)
```bat
mkdir models
copy report_figures\vision\yolo_best.pt models\
copy report_figures\vision\layout_cnn_resnet18.pt models\
streamlit run demo\app.py
```
Pick a layout and resume, choose the quality tier, and press **Run all models**. Preprocessing, OCR, the NLP-only parser, the CNN and YOLO run live. LayoutLMv3 shows its recorded test-run output from Kaggle.

### B. Reproduce the results tables from the saved predictions
The scoring below needs the full dataset (step C).
```bat
python score.py --data dataset --pred preds\nlp_only --pred preds\text_only --pred preds\layout_aware --split test
python make_figures.py --data dataset --preds preds --ocr ocr --out report_figures\figures
```

### C. Full pipeline from scratch
```bat
python generate.py --n 300 --out dataset --workers 4                       :: 1. dataset (~25 min)
python pipeline\run_ocr.py --data dataset --out ocr --workers 4            :: 2. preprocessing + OCR
python pipeline\eval_ocr.py --data dataset --ocr ocr                       :: 3. OCR accuracy
python pipeline\model_nlp_only.py --data dataset --ocr ocr --out preds\nlp_only   :: 4. Model A
python pipeline\layoutlm_prep.py --data dataset --ocr ocr --out layoutlm_data --check  :: 5. LayoutLMv3 data + ceiling
::   6. Upload layoutlm_data to Kaggle, run train_layoutlmv3_kaggle.ipynb and ablation_text_only_kaggle.ipynb,
::      then download the preds\layout_aware and preds\text_only folders
python pipeline\vision_prep.py --data dataset --out vision_data            :: 7. CNN/YOLO data
::   8. Run train_vision_kaggle.ipynb on Kaggle (GPU)
python score.py --data dataset --pred preds\nlp_only --pred preds\text_only --pred preds\layout_aware --split test
python make_figures.py --data dataset --preds preds --ocr ocr --out report_figures\figures
```

---

## 3. Key results (test set, unseen templates)

| | NLP-only | Text-only LayoutLMv3 | Layout-aware LayoutLMv3 |
|---|---|---|---|
| Mean F1, all 12 cells | 0.406 | 0.642 | **0.708** |
| Multi-column, clean | 0.346 | 0.756 | **0.936** |
| Single-column, clean | 0.846 | 0.947 | **0.966** |

- Layout-aware is best in all 12 layout × quality cells, and significantly beats NLP-only in all 12 (Wilcoxon, Bonferroni α = 0.0042).
- Layout information itself (layout-aware vs text-only) helps significantly only on **multi-column** and **table** resumes.
- About 78% of the gain comes from learning and about 22% from layout.
- Heavy degradation cuts OCR word accuracy to 0.59–0.69. This is the main bottleneck for every model.
- ResNet-18 layout classifier: accuracy 0.902. YOLO11 region detector: mAP50 0.731 on unseen templates.

---

## 4. Notes
- The trained LayoutLMv3 weights stayed on Kaggle (too large and could not be exported), so its saved test predictions in `preds/` are provided instead. Both Kaggle notebooks are included to retrain it.
- `.venv`, the full dataset, intermediate OCR files and Kaggle data files are left out to keep the package small. All can be regenerated with the commands above.
