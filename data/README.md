# Data

The raw survey export is **not distributed** with this repository because it
contains individual student responses (N = 209) collected under a
confidentiality/anonymity commitment. To reproduce the analyses, place the
survey file at the **repository root** with the exact name:

```
XLS data.csv
```

All scripts resolve the data via a relative path (`../XLS data.csv` from
`src/`, `../../XLS data.csv` from `src/first/`), so no code changes are needed
once the file is in place.

## Expected format

- CSV export of the Google Forms questionnaire (four sections; 46 items).
- **The header spans two rows**; the scripts read with `header=1`
  (i.e. the *second* row holds the working column labels). Preserve this when
  re-exporting.
- One row per respondent.

## Key columns referenced by the scripts (matched by label substring)

| Purpose | Script | Column identified by |
| --- | --- | --- |
| Engagement / perceived academic impact (binarised 4–5 = High) | `first/predict_engagement.py` | contains `impact` |
| Training-workshop attendance (classification target) | `predictive_analysis.py` | `attending workshops or training sessions` |
| Ethical concern — academic integrity (regression target) | `predict_ethical_concerns.py` | `academic integrity` |
| AI familiarity (PEOU proxy) / frequency of use | multiple | contains `familiar` / `frequency` |

If a target label is not found, scripts fall back to the last column and print
a warning — check the console output to confirm the intended target was used.
