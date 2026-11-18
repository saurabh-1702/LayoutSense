# Model B - Layout-Aware (LayoutLMv3)

Same OCR words, same raster word order as Model A, PLUS each word's box and the page image.

## 1. Package the data (local, ~3-5 min)
    python pipeline\layoutlm_prep.py --data dataset --ocr ocr --out layoutlm_data --check
    python score.py --data dataset --pred preds\label_ceiling --split test      (optional sanity check)
`--check` decodes the *perfect* labels: the best LayoutLMv3 could ever score with this OCR (~1.0 on clean/mild).
Zip the folder: right-click layoutlm_data -> Send to -> Compressed (zipped) folder.

## 2. Train + predict on Kaggle (~30-50 min on a T4)
1. kaggle.com -> Datasets -> New Dataset -> upload layoutlm_data.zip -> Private -> Create.
2. kaggle.com -> Code -> New Notebook -> File -> Import Notebook -> pick
   layoutlm_data\train_layoutlmv3_kaggle.ipynb.
3. Right panel: Add Input -> your layoutlm_data dataset.
   Settings: Accelerator = GPU T4 x2 (or P100), Internet = On (to download microsoft/layoutlmv3-base).
   (Internet needs a phone-verified Kaggle account.)
4. Run All. Watch the per-epoch log: val_token_f1 should rise to ~0.85-0.95.
   If the session stops, just Run All again - it resumes from the last checkpoint.
5. Output panel (right side, /kaggle/working): download preds_layout_aware.zip and training_log.csv.

## 3. Compare (local)
Extract preds_layout_aware.zip into the preds folder (you get preds\layout_aware\clean|mild|heavy\...), then:
    python score.py --data dataset --pred preds\nlp_only --pred preds\layout_aware --split test
