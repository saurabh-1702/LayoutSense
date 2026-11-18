# Stage 1 + OCR (shared by both branches)

## One-time: install Tesseract on Windows
1. Download the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki
   (file like `tesseract-ocr-w64-setup-5.x.x.exe`).
2. Install with the defaults -> `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   (The script finds it there automatically; otherwise pass `--tesseract "<path>"`.)
3. In the activated venv: `pip install pytesseract`

## Run
    python pipeline\run_ocr.py --data dataset --out ocr --split test --tiers heavy --limit 5 --workers 2   (2-minute test)
    python pipeline\run_ocr.py --data dataset --out ocr --workers 4                                         (everything)
    python pipeline\eval_ocr.py --data dataset --ocr ocr

`run_ocr.py` skips images already done, so you can stop it (Ctrl+C) and resume later.

## Output (ocr/tesseract/<tier>/RXXXX_pN.json)
- `words[]`: `text`, `conf`, `bbox` (on the preprocessed page), `bbox_orig` / `poly_orig` (on the original
  image = same coordinates as dataset/annotations), `line_id` (Tesseract reading order), block/par/line ids
- `prep_image`: the preprocessed page (ocr/tesseract/prep/<tier>/RXXXX_pN.jpg). Feed THIS image + `bbox` to LayoutLMv3
- `M`: 3x3 matrix original -> preprocessed;  `preprocess.steps`: what was applied;  `text_blocks`: morphology+contour boxes

## Preprocessing applied
- all tiers: grayscale, Otsu binarisation, morphology + contours (text blocks)
- mild/heavy: page detection + perspective rectification (if the page edge is visible), Hough deskew,
  non-local-means denoising, CLAHE; heavy also illumination flattening (shadow removal)

## OCR evaluation (ocr/ocr_eval_summary.csv, ocr_eval_per_image.csv)
- detection recall / word accuracy / CER per layout x tier (punctuation-only glyphs excluded)
- field ceilings: share of field values OCR read exactly (spaces/case ignored) = the best any parser can reach
  with this OCR. Report these next to model F1 so OCR errors are not blamed on either model (spec section 15).

# Model A - NLP-only (layout-blind)

One-time:  pip install spacy   then   python -m spacy download en_core_web_sm
(optional, slower, closer to the spec's "transformer-based" model:  python -m spacy download en_core_web_trf
 and add  --spacy-model en_core_web_trf)

    python pipeline\model_nlp_only.py --data dataset --ocr ocr --out preds\nlp_only
    python score.py --data dataset --pred preds\nlp_only --split test

Writes preds\nlp_only\<tier>\RXXXX.json (answer-key schema) and preds\nlp_only\_flat\<tier>\RXXXX.txt
(the flattened text the parser saw - good material for the report's error gallery).

Ablation (optional): --order tesseract --out preds\nlp_tess_order uses Tesseract's own block reading order
instead of pure raster order. Tesseract's page segmentation already undoes some column interleaving,
which is worth reporting as a separate row (it is a third, "partly layout-aware OCR" condition).
