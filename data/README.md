# Data

`survey_anonymised.csv` is an **anonymised** version of the Google Forms
survey export (N = 209 respondents, 46 items, four sections). The pipeline
reads it directly, so no setup is needed. The raw export (`XLS data.csv`)
is still **not distributed** and is git-ignored.

## Format

- One header row (read with `header=0`), one row per respondent.
- All 46 columns are kept in their original order, because several scripts
  select columns by position (for example `TARGET_IDX = 32` in
  `src/first/predict_engagement.py`).

## Anonymisation applied

| Column(s) | Change |
| --- | --- |
| `Timestamp` (col 0) | Blanked |
| The four "If other, please specify…" free-text columns (cols 5, 8, 11, 27) | Blanked |
| `Year of Study` (col 2) | Anything other than Freshman / Sophomore / Junior / Senior became `Other` |
| Other single-choice columns | Free-text answers given instead of a listed option (not a Likert value like `4- Agree` and occurring fewer than 3 times) became `Other` |
| Select-all-that-apply columns (cols 4, 7, 10, 26) | Options chosen fewer than 5 times (typed-in "other" entries) became `Other` |

All Likert ratings and standard multiple-choice answers are unchanged.

## Key columns referenced by the scripts

| Purpose | Script | Column identified by |
| --- | --- | --- |
| Engagement / perceived academic impact (binarised 4–5 = High) | `first/predict_engagement.py` | column index 32 |
| Training-workshop attendance (classification target) | `predictive_analysis.py` | `attending workshops or training sessions` |
| Ethical concern — academic integrity (regression target) | `predict_ethical_concerns.py` | `academic integrity` |
| AI familiarity (PEOU proxy) / frequency of use | multiple | contains `familiar` / `frequency` |

If a target label is not found, scripts fall back to the last column and print
a warning — check the console output to confirm the intended target was used.
