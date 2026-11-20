# Idea 1 Results: Predicting Student Academic Engagement & AI Impact

## Overview

**Research Question:** Can student-level features (demographics, AI tool usage, familiarity, usage frequency, task types, and perceived benefits) predict whether a student experiences a **High** or **Low** academic impact from AI usage in their literature studies?

**Target Variable:** Perceived academic impact of AI (`4- Moderate impact`, column 32)  
**Binarisation:** Likert scores 4–5 → *High Impact*; 1–3 → *Low Impact*  
**Sample:** N = 207 (after dropping one missing response)  
**Class Balance:** High Impact = 101 (48.8%), Low Impact = 106 (51.2%) — near-perfectly balanced

**Features Used (13 predictors):**

| Feature | Column | Description |
|---|---|---|
| Gender | Male | Binary gender indicator |
| Year of Study | Senior | Academic year (Senior/Junior/etc.) |
| Literature Courses | More than 6 | Number of lit. courses completed |
| AI Familiarity | Familiar | Self-rated AI familiarity level |
| Primary AI Tool | ChatGPT | Primary tool used |
| Secondary Tool | Unnamed:8 | Secondary tool |
| Usage Frequency | Occasionally | How often AI is used |
| Primary Task | Analyzing literary texts | Main task AI is used for |
| Secondary Task | Unnamed:11 | Secondary task |
| Tertiary Task | Comparative lit. essays | Task type |
| Perceived Benefit 1 | 5- Strongly agree | Agreement with benefit statement 1 |
| Perceived Benefit 2 | 4- Agree | Agreement with benefit statement 2 |
| Perceived Benefit 3 | 4- Agree.1 | Agreement with benefit statement 3 |

---

## Mathematical Foundations

### Binarisation of the Target

The raw Likert score $y_i \in \{1,2,3,4,5\}$ is transformed into a binary class label:

$$\hat{y}_i = \begin{cases} \text{High Impact} & \text{if } y_i \geq 4 \\ \text{Low Impact} & \text{if } y_i < 4 \end{cases}$$

This preserves interpretability: students who "Agree" or "Strongly Agree" that AI has a positive academic impact form the positive class.

### Algorithm Mathematical Bases

**Random Forest** — splits nodes to minimise Gini Impurity:
$$H(\mathcal{R}) = 1 - \sum_{k} \hat{p}_k^2$$

**SVM (RBF Kernel)** — maximises the margin:
$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(\mathbf{w}^\top\phi(\mathbf{x}_i)+b) \geq 1-\xi_i$$

with kernel $K(\mathbf{x},\mathbf{z})=\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$.

**Gradient Boosting** — additive model via pseudo-residuals:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\,\gamma_m h_m(\mathbf{x}),\quad r_{im}=-\frac{\partial \mathcal{L}(y_i,F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}$$

**k-NN (k=7)** — Euclidean distance vote:
$$d(\mathbf{x}_i,\mathbf{x}_j)=\sqrt{\sum_{l=1}^{p}(x_{il}-x_{jl})^2}$$

**Naive Bayes (Gaussian)** — Bayes' theorem under independence:
$$P(y|\mathbf{x}) \propto P(y)\prod_{j=1}^{p}\frac{1}{\sqrt{2\pi\sigma_{jy}^2}}\exp\!\left(-\frac{(x_j-\mu_{jy})^2}{2\sigma_{jy}^2}\right)$$

**Logistic Regression** — sigmoid probability, L2-regularised:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}+b),\quad \mathcal{L}=-\sum_i \log P(y_i|\mathbf{x}_i)+\lambda\|\mathbf{w}\|^2$$

---

## Results

### Algorithm Performance Comparison (Test Set + 5-Fold CV)

| Algorithm | Accuracy | F1 (W) | Precision | Recall | CV Mean Acc | CV Std |
|---|---|---|---|---|---|---|
| **Naive Bayes** | **0.6905** | **0.6907** | **0.6912** | **0.6905** | **0.6131** | 0.0757 |
| SVM (RBF) | 0.5952 | 0.5955 | 0.5956 | 0.5952 | 0.5552 | 0.0594 |
| k-NN (k=7) | 0.5952 | 0.5917 | 0.5944 | 0.5952 | 0.5796 | 0.0723 |
| Logistic Regression | 0.5952 | 0.5955 | 0.5956 | 0.5952 | 0.5839 | 0.0834 |
| Random Forest | 0.5476 | 0.5468 | 0.5479 | 0.5476 | 0.6128 | 0.0972 |
| Gradient Boosting | 0.5238 | 0.5216 | 0.5222 | 0.5238 | 0.5497 | 0.0956 |

**Key insight:** **Naive Bayes achieved the highest test accuracy (69.05%) and balanced F1 score (0.691)**, outperforming all ensemble and kernel methods. This is a notable and theoretically interesting finding — despite its strong conditional independence assumption (known to be violated given the correlated benefit items, r ≈ 0.61), Naive Bayes generalises best. This is consistent with literature showing that Naive Bayes benefits from limited training data (n = 165 training samples) due to its extremely low variance, making it the preferred model for this task.

### Figure: Algorithm Comparison
![Algorithm Comparison](./Figure_AlgorithmComparison.png)

### Figure: ROC Curves (AUC)
![ROC Curves](./Figure_ROC.png)

### Figure: 5-Fold CV per Fold
![CV Folds](./Figure_CV_Folds.png)

---

## Feature Importance (Random Forest — MDI)

The **Mean Decrease in Impurity** reveals which features most strongly separate High-Impact from Low-Impact students:

$$\text{MDI}(j) = \sum_{t=1}^{T}\sum_{s \in \mathcal{S}_t^{(j)}} \frac{|\mathcal{R}_s|}{n} \cdot \Delta H(s)$$

### Figure: Feature Importance
![Feature Importance](./Figure_FeatureImportance.png)

The **perceived benefit items** (agreement with AI benefit statements) dominate the feature importance ranking, significantly outweighing demographic and tool-usage features. This is pedagogically crucial: it implies that a student's subjective conviction about AI's utility — rather than *how* or *how often* they use it — is the primary determinant of whether they experience high academic impact.

---

## Statistical Significance: High vs Low Impact Groups (Mann-Whitney U)

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1, \quad r = 1 - \frac{2U}{n_1 n_2}$$

| Feature | U | p-value | Effect Size r | Significant |
|---|---|---|---|---|
| 4- Agree (Benefit 2) | 7391 | **< .0001** | **0.381** | ✅ Yes |
| 5- Strongly Agree (Benefit 1) | 7078 | **< .0001** | **0.322** | ✅ Yes |
| 4- Agree.1 (Benefit 3) | 7072 | **< .0001** | **0.321** | ✅ Yes |
| Usage Frequency | 4728 | .125 | 0.117 | ❌ No |
| Year of Study | 4742 | .122 | 0.114 | ❌ No |
| Gender | 5118 | .488 | 0.044 | ❌ No |
| AI Familiarity | 5540 | .650 | 0.035 | ❌ No |
| AI Tool Used | 5348 | .991 | 0.001 | ❌ No |

The three **perceived benefit agreement items** are the only statistically significant features (all p < .0001), with medium effect sizes (r = 0.32–0.38). Gender, tool choice, familiarity, and usage frequency show no significant relationship with engagement outcome — a finding that challenges technological determinism and suggests that psychological disposition toward AI utility is the primary lever for pedagogical intervention.

---

## PCA Projection

### Figure: PCA — True vs SVM Predicted Labels
![PCA Projection](./Figure_PCA.png)

PCA confirms that High and Low Impact students **overlap substantially** in the two-dimensional projection (PC1 explains 29.0% variance), which is consistent with the moderate classifier accuracies (55–69%). The feature space does not produce linearly separable clusters — reinforcing why Naive Bayes (which estimates Gaussian class-conditional densities in the full space) outperforms distance-based methods like k-NN in this setting.

---

## Target Distribution

### Figure: Likert Score Distribution by Engagement Group
![Distribution](./Figure_Distribution.png)

---

## Confusion Matrices

| Random Forest | SVM (RBF) |
|---|---|
| ![CM RF](./CM_Random_Forest.png) | ![CM SVM](./CM_SVM_RBF.png) |

| Naive Bayes | k-NN |
|---|---|
| ![CM NB](./CM_Naive_Bayes.png) | ![CM kNN](./CM_k-NN_k=7.png) |

| Gradient Boosting | Logistic Regression |
|---|---|
| ![CM GBC](./CM_Gradient_Boosting.png) | ![CM LR](./CM_Logistic_Regression.png) |

---

## Summary of Idea 1 Findings

| Aspect | Finding |
|---|---|
| **Best Algorithm** | Naive Bayes — Test Acc = 69.05%, F1 = 0.691 |
| **Best CV Algorithm** | Naive Bayes — CV Acc = 61.31% ± 7.57% |
| **Top Predictor** | Agreement with perceived AI benefits (effect r = 0.38, p < .0001) |
| **Non-significant** | Gender, AI tool choice, familiarity, usage frequency |
| **Class Balance** | Near-perfectly balanced (48.8% High, 51.2% Low) |
| **Key Insight** | Attitudinal belief in AI benefits — not behaviour — drives academic impact |

### Implications

1. **Engagement is attitudinally mediated**: The strongest predictor of positive academic impact is not *what tool* a student uses or *how often*, but their *belief* that AI is beneficial. This implies instructors should focus on **attitude-formation interventions** (workshops demonstrating concrete literary analysis benefits) rather than purely skills-based training.
2. **Naive Bayes outperforms complex ensembles**: In small-n educational survey datasets, simpler probabilistic models with low variance often outperform high-variance ensemble methods — a consistent pattern in educational data mining literature.
3. **Moderate prediction ceiling**: The best CV accuracy of ~61% suggests that engagement is a genuinely complex, multi-determined outcome not fully captured by 13 survey features. Future work incorporating longitudinal grade data or open-text analysis could improve predictive power.
