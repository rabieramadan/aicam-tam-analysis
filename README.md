# GenAI Adoption in English Literature Education — Analysis Code

Reproducible analysis pipeline for a **Technology Acceptance Model (TAM)** study
of Generative AI adoption among undergraduate English Literature students at
Sultan Qaboos University (N = 209). The code covers supervised prediction,
unsupervised student profiling, association-rule mining, and non-parametric
statistical testing, and regenerates the figures and tables used in the paper.

> **Data notice.** The raw survey file (`XLS data.csv`) is confidential and is
> **not** included. Place your own copy at the repository root to reproduce the
> results — see [`data/README.md`](data/README.md).

## Repository layout

```
.
├── src/
│   ├── predictive_analysis.py       # Classification: RF, SVM-RBF, Gradient Boosting, LogReg (+ MDI importance)
│   ├── advanced_analysis.py         # Stats tests + Naive Bayes/k-NN, ROC, confusion matrices, Bayesian opt, ANOVA
│   ├── student_profiling.py         # Clustering (K-Means/Agglomerative), PCA, dendrogram, cluster validation
│   ├── predict_ethical_concerns.py  # Regression: Random Forest predicting academic-integrity concern
│   ├── generate_journal_assets.py   # Paper-ready figures + Apriori association rules (Table 4)
│   ├── compile_report.py            # Assembles result tables into one Markdown report (run last)
│   └── first/
│       └── predict_engagement.py    # "Idea 1": predicts perceived academic impact (High/Low)
├── data/
│   └── README.md                    # Data schema + placement instructions (data itself is git-ignored)
├── results/
│   ├── figures/                     # Example output figures (PNG)
│   └── tables/                      # Example output tables (CSV)
├── docs/                            # Narrative result write-ups (Complete_Results.md, etc.)
├── run_all.py                       # Runs the full pipeline end to end
├── requirements.txt
├── LICENSE                          # MIT
└── CITATION.cff
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/rabieramadan/aicam-tam-analysis.git
cd aicam-tam-analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Put the survey export at the repo root as `XLS data.csv` (two-row header;
   scripts read with `header=1`). See [`data/README.md`](data/README.md).
2. Run everything:

   ```bash
   python run_all.py
   ```

   …or run any stage on its own from the repo root, e.g.:

   ```bash
   python src/predictive_analysis.py
   python src/student_profiling.py
   python src/first/predict_engagement.py
   ```

Each script prints metrics to the console and writes its figures/tables to its
working directory. Run `compile_report.py` **after** the table-producing
scripts so it can gather their CSV outputs.

## What each script does

| Script | Type | Key outputs |
| --- | --- | --- |
| `predictive_analysis.py` | Supervised classification of a training/attendance target across four algorithms with weighted accuracy/F1/precision/recall/AUC and MDI feature importance | RF metrics, algorithm-comparison table, feature-importance figure |
| `advanced_analysis.py` | Chi-square, Kruskal–Wallis and Mann–Whitney U tests; adds Naive Bayes and k-NN; one-vs-rest ROC; confusion matrices; optional Bayesian hyperparameter search; Likert correlation heatmap; cluster-wise ANOVA | Significance tables, ROC curves, confusion matrices, correlation heatmap |
| `student_profiling.py` | Unsupervised profiling: optimal-k selection (elbow, silhouette, Davies–Bouldin, Calinski–Harabasz), K-Means and Agglomerative clustering, PCA, dendrogram | Cluster-validation table, PCA-variance table, cluster/dendrogram plots |
| `predict_ethical_concerns.py` | Random Forest **regression** predicting the academic-integrity ethical-concern score, with importance ranking | Ethical-concern importance plot |
| `generate_journal_assets.py` | Builds paper-ready demographics/descriptives figures and mines usage patterns with the Apriori algorithm (min support 0.15, confidence 0.70) | Demographics/benefits figures, descriptives table, association-rules table |
| `first/predict_engagement.py` | Binarises perceived academic impact (4–5 = High) and benchmarks six classifiers with cross-validation | Confusion matrices, ROC, PCA, algorithm-comparison figures/tables |
| `compile_report.py` | Reads the generated CSV tables and emits a single consolidated Markdown report | `Complete_Results.md` |

## Reproducibility notes

- Random seeds are fixed (`random_state=42`) in the model scripts.
- `scikit-optimize` is optional; without it, `advanced_analysis.py` skips the
  Bayesian search and continues.
- Target columns are matched by label substring; if a match fails the script
  falls back to the last column and prints a warning — check the console to
  confirm the intended target.

## License

MIT — see [`LICENSE`](LICENSE).
