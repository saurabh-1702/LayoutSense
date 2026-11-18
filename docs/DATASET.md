# Synthetic Resume Dataset Generator

Generates realistic resume **images** in 4 layout types × 3 image-quality tiers, each with a
complete **answer key**, so the NLP-only and Layout-Aware pipelines can be scored against
exactly the same truth.

* **Layouts** (3 templates each, 12 total):
  `single_column` (s1_classic, s2_modern, s3_banded), `multi_column` (m1_sidebar_left,
  m2_sidebar_right, m3_two_equal), `table_based` (t1_grid, t2_cv_table, t3_form),
  `infographic` (i1_dark_sidebar, i2_banner_cards, i3_timeline_gauges). No photos anywhere.
* **Tiers**: `clean`, `mild`, `heavy`, using the composite degradation in spec §6.4
  (rotation + perspective warp + noise + blur + brightness/contrast + JPEG; heavy also gets an uneven-lighting shadow).
* **Content**: fictional people with realistic names from 50 country/region pools (India split into
  North/South/East/West), phone numbers in each country's real mobile format, real universities, employers,
  certifications and hospitals, 20 industries, freshers to 20+ years, 1- and 2-page resumes.
  No placeholders such as "John Doe" or "98XXXXXXXX".

## Setup (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

Tesseract and Poppler are **not** needed to generate the dataset (only for your pipelines).

## Generate

```bat
python generate.py --n 300 --out dataset --workers 4
```

* About 4 s per resume per worker (≈ 20–25 min for 300 resumes on 4 workers).
* Disk: about 5 MB per resume (all tiers and pages), so ≈ 1.5 GB for 300 resumes. `--degraded-png` stores
  mild/heavy as PNG instead (≈ 25 GB for 300; see "Image formats").
* Same `--seed` always gives the same dataset. Re-build single resumes with `--only R0007,R0019`.
* Fonts: templates ask for Calibri, Cambria, Arial, Georgia, Segoe UI, etc., which Windows has. The sample
  batch was rendered on Linux with metric-compatible substitutes, so your images will look slightly
  different. **The answer key is always measured from the actual render**, so it stays exact either way.

## Output

```
dataset/
  manifest.csv                     one row per image (see columns below)
  images/clean/R0001_p1.png        300 DPI, A4 (2481×3509) or US Letter (2550×3300)
  images/mild/R0001_p1.jpg         degraded; canvas grows to fit the rotated page
  images/heavy/R0001_p1.jpg
  ground_truth/R0001.json          FIELD-LEVEL ANSWER KEY (same for all tiers and pages)
  text/R0001_p1.txt                gold text of that page, in reading order
  text/R0001.txt                   gold text of the whole resume
  annotations/{tier}/R0001_p1.json word / line / region boxes + labels, in THAT image's coordinates
  yolo/labels/{tier}/R0001_p1.txt  region boxes in YOLO format;  yolo/classes.txt
  html/R0001.html                  source HTML (debugging / re-rendering)
```

### ground_truth/RXXXX.json — what each pipeline is scored against

Only what is **printed on the page** is included. If a template doesn't show a candidate's job location,
that key is absent. Values are the exact displayed strings, plus normalised extras for lenient matching:

```json
{
  "resume_id": "R0007",
  "name": "AHMED MOSTAFA MAHMOUD",
  "contact": {"email": "...", "phone": "+20 118 735 8997", "location": "Giza, Egypt",
              "phone_e164": "+201187358997", "city": "Giza", "country": "Egypt"},
  "links": [{"url": "linkedin.com/in/...", "type": "LinkedIn"}],
  "education": [{"degree": "BS, Information Technology", "institution": "Nile University",
                 "dates": "Aug 2011–Aug 2015", "grade": "GPA: 3.40",
                 "start": "2011-08", "end": "2015-08", "level": "bachelor"}],
  "experience": [{"title": "...", "company": "...", "location": "...", "dates": "Nov 2025–Present",
                  "bullets": ["..."], "start": "2025-11", "end": "present"}],
  "skills": ["Redis", "..."],
  "projects": [{"name": "...", "description": "...", "tech": "..."}],
  "certifications": [{"name": "...", "issuer": "...", "year": "2023"}],
  "awards": [{"name": "...", "issuer": "...", "year": "2021"}],
  "languages": [{"language": "Arabic", "proficiency": "Native"}],
  "_meta": {"layout": "table_based", "template": "t2_cv_table", "n_pages": 2, "industry": "software", "level": "senior"}
}
```

Notes for scoring:
* Some templates print the name in capitals; compare **case-insensitively**.
* Table template `t3_form` prints experience dates as separate *From* / *To* columns, so those entries have
  `start_text` / `end_text` instead of `dates`. Use the normalised `start` / `end` to compare dates across all layouts.
* `education[].year` appears where a template prints only the year of passing.

### annotations/{tier}/RXXXX_pN.json — token-level truth (for LayoutLMv3, YOLO, overlays, OCR error analysis)

* `words[]`: `text`, `bbox` [x0,y0,x1,y1] in pixels, `poly` (4 corners, follows rotation/warp), `field`
  (e.g. `experience.0.company`), `label` (BIO tag), `line`. Words are OCR-style: punctuation touching a word
  is merged into it (e.g. `Python,`). A word that wraps across lines (e.g. `board-` / `level`) is split per line.
* BIO labels: `NAME, HEADLINE, EMAIL, PHONE, LOCATION, LINK, EDU_DEGREE, EDU_INSTITUTION, EDU_DATE, EDU_GRADE,
  EDU_LOCATION, EXP_TITLE, EXP_COMPANY, EXP_DATE, EXP_LOCATION, EXP_DESC, SKILL, SKILL_CATEGORY, SKILL_LEVEL,
  PROJ_NAME, PROJ_DESC, PROJ_TECH, PROJ_DATE, CERT_NAME, CERT_ISSUER, CERT_DATE, AWARD_NAME, AWARD_ISSUER,
  AWARD_DATE, LANGUAGE, LANG_LEVEL, SECTION_HEADING`, everything else `O`.
  For LayoutLMv3, normalise boxes to 0–1000 with `x*1000/width`, `y*1000/height`.
* `lines[]`, `regions[]` (`header, sidebar, column, table, section_*`), `degradation` (exact parameters used)
  and `homography` (3×3 matrix from the clean page to this image; `null` for clean).

### Gold text and reading order

`text/*.txt` holds what a perfect reader would produce: header first, then each column top-to-bottom,
**left column fully before the right column**. This is the logical order the NLP-only branch loses when OCR
reads straight across columns. Bullet glyphs (•, –, ▪) and table header labels are included because they are on the page.

### Splits (manifest `split` column)

`test` = the 4 held-out templates (`s3_banded, m3_two_equal, t3_form, i3_timeline_gauges`) which **never
appear in train/val**. That stops a fine-tuned LayoutLMv3 or YOLO model from scoring well just by memorising where fields sit
on a familiar template. Remaining resumes: ≈85% `train`, 15% `val`. All tiers and pages of one resume share a split.
Change `TEST_TEMPLATES` in `generate.py` to choose other held-out templates.

### manifest.csv columns

`image, resume_id, page, n_pages, tier, layout, template, split, industry, level, years_experience,
origin_country, work_country, page_size, width, height, n_words, font, font_px, ground_truth, annotation,
gold_text, degradation`

### Image formats

Clean pages are lossless PNG. The last step of the mild/heavy degradation is JPEG compression (quality 70 / 30,
as in spec §6.4). Those pages are saved as **that exact JPEG**, which is identical to the degraded image, not a
second compression. Saving them as PNG gives the same pixels but noise-filled PNGs are 30–45 MB each.

## Score the two pipelines

Have each pipeline write one JSON per resume per tier, in the same schema as `ground_truth` (omit fields it
doesn't extract):

```
preds/nlp_only/clean/R0001.json   preds/nlp_only/mild/R0001.json   preds/nlp_only/heavy/R0001.json
preds/layout_aware/clean/R0001.json ...
```

```bat
python score.py --data dataset --pred preds\nlp_only --pred preds\layout_aware --split test
```

Prints F1 per field (name, email, phone, location, education, experience, skills, certifications, languages,
projects) for every layout × tier cell, plus a Wilcoxon signed-rank test per cell on paired per-resume F1 (spec §10.1).
It also writes `scores_per_resume.csv` with the fuzzy-match ratio per field (≥ 90 green / 50–89 yellow / < 50 red, spec §11.1).

## Check alignment visually (spec §6.6 step 8)

```bat
python tools\visualize.py dataset annotations\heavy\R0007_p1.json check.png            :: words coloured by label
python tools\visualize.py dataset annotations\heavy\R0007_p1.json check.png --regions  :: layout regions
```

## Tuning

* Seniority mix and share of 2-page resumes: `LEVELS`, `TWO_PAGE_P` in `generate.py`.
* Degradation ranges: `TIERS` in `resumegen/degrade.py`.
* Names, phone formats, universities, employers: `resumegen/data_regions.py`.
  Industries, titles, skills, bullet templates, certifications: `resumegen/data_industries.py`.
* Visual variation (fonts, colours, heading synonyms, section order, date formats): `resumegen/style.py`, `resumegen/content.py`.
* Templates: `resumegen/templates/*.html.j2`. Wrap every printed value in an element with `data-field="<path>"`.
  That's how the answer key is built.

## Data ethics

Every person is fictional: names are random combinations of common first and last names, and phone numbers are
random digits in real national formats. Some numbers may therefore exist. Use the dataset for research and
don't present the resumes as real people.
