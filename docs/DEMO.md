# Live demo: one resume, every model, side by side

Pick a resume (or upload one) → press **Run all models** → the app runs, live on your PC:

| Stage | What runs |
|---|---|
| CV preprocessing | grayscale, Otsu, morphology, contours (+ page detection, deskew, denoise, CLAHE for mild/heavy) |
| OCR | Tesseract 5, same settings as the experiments |
| NLP-only parser | raster flattening → regex + spaCy + rules |
| LayoutLMv3 text-only & layout-aware | the trained models' outputs from the Kaggle test run (the exact predictions reported in the paper), for the dataset's test resumes |
| CNN layout classifier | ResNet-18 |
| YOLO region detector | YOLO11-s |

Seven tabs: **(1)** the resume, its preprocessed version, YOLO regions and the CNN verdict · **(2)** what the NLP model read
(flattened lines, words coloured by their true section, ⚠️ on lines that mix two columns) · **(3)** what LayoutLMv3 read (the same
lines, coloured by the field LayoutLMv3 predicted, wrong words struck through) · **(4)** the **full parsed resume from every
model** side by side with the answer key — each item green/yellow/red, plus a *Missed* list · **(5)** field-by-field scores ·
**(6)** each model's labels painted on the page with wrong words framed red, plus a list of every mistake · **(7)** whole-test-set results.

## 1. Install (once)
```
cd /d D:\projectss\resume_synth
.venv\Scripts\activate
pip install -r demo\requirements_demo.txt
```

## 2. Put the model files in place
```
mkdir models
copy report_figures\vision\yolo_best.pt models\
copy report_figures\vision\layout_cnn_resnet18.pt models\
```
(The LayoutLMv3 outputs are read from `preds\layout_aware` and `preds\text_only`, which you already have.)

## 3. Run
Double-click `demo\run_demo.bat`, or:
```
streamlit run demo\app.py
```
The browser opens at http://localhost:8501. The first run loads the models (~30–60 s); after that each resume takes
~10–30 s (mostly OCR, more for heavy images because of denoising).

## Suggested 3-minute demo for the invigilator
1. **Multi-column, clean** → Run. Tab 2: point at the ⚠️ lines ("Frontend Engineer Dublin City University").
   Tab 3: the same jumbled lines, but LayoutLMv3 colours almost every word correctly.
   Tab 4: the full parsed resume of each model side by side — NLP-only's jobs glued across columns (red), layout-aware matching the answer key (green).
2. Same resume, **heavy** → Run. Tab 1: preprocessing straightens the photo; YOLO still finds every region; scores drop for all parsers (OCR is the bottleneck).
3. **Single-column, clean** → Run. Tab 4: all three parsed resumes look almost the same — layout has nothing to fix.
4. **Table-based** or **Infographic** → Run. Tab 6: where each model mislabels words.
5. Tab 7: the overall result table and heatmaps.

Note: uploaded images (not from the dataset) run preprocessing, OCR, the NLP-only parser, the CNN and YOLO live; LayoutLMv3
outputs exist only for the dataset's test resumes, because the trained LayoutLMv3 weights stayed on Kaggle.
