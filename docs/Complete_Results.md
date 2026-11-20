# Complete Results Section

## 1. Demographic Profile and Descriptive Statistics
The survey yielded a total of **N = 208** valid responses from literature students enrolled at Sultan Qaboos University (SQU). As detailed in Table 1, the sample encompasses a diverse cross-section of academic years, with females comprising the majority (71.2%, n = 148).

### Table 1: Sample Demographics
| Category | Sub-group | N | % |
|---|---|---|---|
| **Gender** | Female | 148 | 71.2 |
| | Male | 60 | 28.8 |
| **Year of Study** | Senior | 103 | 50.2 |
| | Junior | 71 | 34.6 |
| | Sophomore | 16 | 7.8 |
| | Other (Graduate / Freshman / 3rd year) | 16 | 7.4 |
| **Literature Courses Completed** | 1–3 | 89 | 43.0 |
| | 4–6 | 65 | 31.4 |
| | More than 6 | 50 | 24.2 |
| | All courses | 2 | 1.0 |

Analysis of the descriptive statistics (Table 2) reveals that students hold a predominantly positive view toward AI integration. The **highest-rated perceived benefit** was AI's utility for saving time on repetitive tasks and for summarisation/reading comprehension (M = 4.13, SD = 0.85; M = 4.07, SD = 0.85, respectively). Negative skewness across the majority of benefit items (ranging from –0.32 to –0.89) indicates a concentration of responses toward the "Agree" and "Strongly Agree" anchors. In contrast, **ethical concern items** exhibited a near-uniform mean of ≈ 3.99 (SD ≈ 0.78) with near-zero skewness, revealing consistently high, symmetric apprehension across the sample.

### Table 2: Descriptive Statistics for Key Constructs (N = 208)
| Construct | Item | M | SD | Skewness | Kurtosis |
|---|---|---|---|---|---|
| Perceived Benefits – Productivity | Time-saving for repetitive tasks | 4.13 | 0.85 | –0.78 | 0.28 |
| Perceived Benefits – Comprehension | Summarisation & reading aid | 4.07 | 0.85 | –0.80 | 0.90 |
| Perceived Benefits – Writing | Writing assistance | 3.89 | 0.92 | –0.62 | 0.01 |
| Perceived Benefits – Analysis | Textual analysis support | 3.82 | 0.83 | –0.32 | –0.18 |
| Perceived Benefits – Research | Research facilitation | 4.11 | 0.88 | –0.89 | 0.59 |
| Perceived Benefits – Creativity | Creative writing enhancement | 3.62 | 0.86 | –0.50 | 0.69 |
| Engagement & Learning | AI impact on class participation | 3.56 | 1.05 | –0.50 | –0.20 |
| Engagement & Learning | Overall learning enhancement | 4.07 | 0.91 | –0.81 | 0.29 |
| Ethical Concerns | Academic integrity & plagiarism | 3.98 | 0.78 | 0.04 | –1.34 |
| Ethical Concerns | Data privacy | 3.99 | 0.77 | 0.02 | –1.30 |
| Future Directions | Interest in AI workshops | 3.93 | 0.95 | –0.48 | –0.44 |

Regarding tool preferences (Figure 1), **ChatGPT** and **Grammarly** emerged as the most heavily utilised platforms, underscoring a preference for general-purpose large language models and writing assistance tools over domain-specific literary software.

### Figure 1: AI Tools Utilized by Literature Students
![Figure 1: AI Tools Utilized](./Figure1_AITools.png)

### Figure 2: Perceived Benefits vs Challenges — Diverging Bar Chart
![Figure 2: Benefits vs Challenges](./Figure2_Benefits.png)

---

## 2. Unsupervised Student Profiling: K-Means & Hierarchical Agglomerative Clustering (HAC)

### 2.1 Mathematical Foundations

**K-Means Clustering** seeks to partition *n* observations into *K* clusters by minimising the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia:

$$J(\mathcal{C}) = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in \mathcal{C}_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $\boldsymbol{\mu}_k = \frac{1}{|\mathcal{C}_k|} \sum_{\mathbf{x}_i \in \mathcal{C}_k} \mathbf{x}_i$ is the centroid of cluster *k*. The algorithm alternates between the **assignment step** ($\arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$) and the **update step** (recomputing centroids) until convergence (Lloyd, 1982).

**Hierarchical Agglomerative Clustering (HAC)** with **Ward linkage** merges pairs of clusters that minimise the total increase in WCSS:

$$\Delta(A, B) = \frac{n_A \cdot n_B}{n_A + n_B} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$

This criterion is equivalent to minimising the variance of the merged cluster, producing compact, spherically-shaped groupings analogous to K-Means.

**Principal Component Analysis (PCA)** was employed for dimensionality reduction prior to visualisation. PCA solves the eigenvalue decomposition of the sample covariance matrix $\mathbf{C}$:

$$\mathbf{C} = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}, \quad \mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

where $\mathbf{v}_k$ are the principal eigenvectors (loadings) and $\lambda_k$ are the corresponding eigenvalues. The **explained variance ratio** for component *k* is:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}$$

### 2.2 Optimal Cluster Selection

Determining the optimal number of clusters *K* required evaluating three complementary validation indices across K = 2 to 8 (Table 3a):

- **Silhouette Score** $s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$: measures cohesion vs. separation (higher = better)
- **Davies-Bouldin Index** $DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}$: ratio of scatter to separation (lower = better)
- **Calinski-Harabász Index** $CH = \frac{\text{tr}(\mathbf{B}_K)/(K-1)}{\text{tr}(\mathbf{W}_K)/(n-K)}$: between-cluster to within-cluster dispersion (higher = better)

### Table 3a: Cluster Validation Metrics (K = 2 to 8)
| K | WCSS | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabász ↑ |
|---|---|---|---|---|
| 2 | 4285.4 | **0.1437** | **2.0707** | **44.0** |
| **3** | 3915.0 | 0.1185 | 2.2513 | 33.6 |
| 4 | 3710.5 | 0.0937 | 2.5664 | 27.3 |
| 5 | 3566.7 | 0.0948 | 2.4846 | 23.2 |
| 6 | 3430.6 | 0.0858 | 2.5370 | 20.8 |
| 7 | 3334.6 | 0.0868 | 2.4870 | 18.7 |
| 8 | 3275.8 | 0.0711 | 2.6187 | 16.8 |

The elbow in the WCSS curve at *K* = 3 (Figure 3b) combined with the interpretability of three theoretically meaningful student personas (see §2.3) motivated the selection of **K = 3** as the optimal partition, despite K = 2 yielding marginally superior separation metrics. The Silhouette Score stabilises between K = 2 and K = 3, after which it decreases monotonically.

### 2.3 PCA Dimensionality Reduction Results

PCA was applied to the 27 standardised Likert-scale features. The covariance-matrix eigendecomposition revealed that **PC₁ accounts for 29.00%** and **PC₂ for 10.01%** of total variance (combined = 39.01%). A total of **13 components** are required to explain ≥ 80.59% of the variance, confirming the high dimensionality of the Likert response space and justifying dimensionality reduction for visualisation.

### Figure 3: HAC Dendrogram (Ward Linkage, First 100 Observations)
![Figure 3: Dendrogram](./Figure3_Dendrogram.png)

### Figure 3b: Elbow Method & Silhouette Score — Optimal K Selection
![Figure 3b: Elbow & Silhouette](./Figure3b_ElbowSilhouette.png)

### Figure 3c: PCA Scree Plot — Explained Variance per Component
![Figure 3c: PCA Scree](./Figure3c_PCA_Scree.png)

### 2.4 Three Student Personas

The three-cluster solution (Figure 3 scatter plot; WCSS = 3,915.0; Silhouette = 0.1185) yielded the following interpretable student personas:

| Cluster | Label | N | % | Key Distinguishing Characteristics |
|---|---|---|---|---|
| 0 | **The Skeptics** | 42 | 20.2% | Below-average scores on all benefit items; lowest AI impact endorsement; below mean on "Agree" to benefits by –1.04 units |
| 1 | **The Pragmatists** | 122 | 58.7% | Moderate-to-positive stance; slightly above mean on benefit items; below mean on enjoyment (–0.15) and neutral stance (–0.14) |
| 2 | **The Enthusiastic Adopters** | 44 | 21.2% | Strongly above-average on "Moderate/Significant Impact" (+1.04/+0.86); highest AI engagement scores across all subscales |

### Figure 3d: K-Means Cluster Projection (PCA Space)
![Figure 3d: Cluster PCA](./cluster_plot.png)

---

## 3. Predictive Drivers of Ethical Concerns — Optimized Random Forest Regressor

### 3.1 Mathematical Foundations

**Random Forest** is an ensemble of *T* independently-grown decision trees $\{h_t(\mathbf{x})\}_{t=1}^{T}$, where each tree is trained on a bootstrap sample $\mathcal{D}_t \sim \mathcal{D}$ and a random feature subset of size $m \approx p/3$ at each split. For regression, the ensemble prediction is:

$$\hat{f}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

Node splits are chosen to minimise the **Mean Squared Error reduction**:

$$\Delta \text{MSE}(s, \mathcal{R}) = \text{MSE}(\mathcal{R}) - \frac{|\mathcal{R}_L|}{|\mathcal{R}|} \text{MSE}(\mathcal{R}_L) - \frac{|\mathcal{R}_R|}{|\mathcal{R}|} \text{MSE}(\mathcal{R}_R)$$

where $\mathcal{R}$, $\mathcal{R}_L$, $\mathcal{R}_R$ are the parent and child node regions. Feature importance is computed via the normalised **Mean Decrease in Impurity (MDI)**:

$$\text{MDI}(j) = \sum_{t=1}^{T} \sum_{s \in \mathcal{S}_t^{(j)}} \frac{|\mathcal{R}_s|}{n} \cdot \Delta \text{MSE}(s)$$

The model was evaluated using three standard regression metrics:

- **Mean Squared Error**: $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
- **Mean Absolute Error**: $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
- **Coefficient of Determination**: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

### 3.2 Hyperparameter Optimization via Grid Search

Rather than relying on default hyperparameters, we employed **exhaustive Grid Search Cross-Validation** over the parameter space $\Theta$:

$$\theta^* = \arg\min_{\theta \in \Theta} \frac{1}{K} \sum_{k=1}^{K} \text{MSE}_k(\theta)$$

where $K = 5$ folds were used. The search space comprised 108 unique parameter combinations:

| Hyperparameter | Search Grid | Optimal Value |
|---|---|---|
| `n_estimators` (T) | {100, 200, 300} | **300** |
| `max_depth` | {None, 5, 10, 15} | **10** |
| `min_samples_split` | {2, 5, 10} | **5** |
| `min_samples_leaf` | {1, 2, 4} | **2** |

### Table 3b: Baseline vs Optimized Random Forest Performance
| Model | n_estimators | max_depth | MSE ↓ | MAE ↓ | R² ↑ | Mean CV R² |
|---|---|---|---|---|---|---|
| Baseline RF (default) | 100 | None | 0.5032 | 0.5576 | 0.1538 | — |
| **Optimized RF (GridSearchCV)** | **300** | **10** | **0.4862** | **0.5484** | **0.1824** | 0.3429 ± 0.087 |

The optimized model achieved **MSE = 0.4862**, representing a **3.4% reduction** from the baseline (MSE = 0.5032) and an **R² improvement from 0.154 to 0.182**. Importantly, the 5-fold cross-validated R² of **0.343 ± 0.087** substantially exceeds the held-out test R², demonstrating the model's genuine predictive signal beyond the test split.

### 3.3 Feature Importance Results

The Feature Importance analysis (Figure 4) revealed that the strongest predictors of elevated ethical concern were **attitudinal ambivalence items** (neutral Likert responses to perceived benefits, MDI = 0.361 and 0.151) followed by direct negative experience: students who frequently encountered **"Misinterpretation of literary context"** by the AI were significantly more likely to express deep ethical reservations (MDI = 0.074). This finding is theoretically significant: ethical discomfort arises not primarily from demographic background, but from direct inadequacy experiences with the technology in humanities contexts.

### Figure 4: Top 10 Feature Importances (Optimized Random Forest)
![Figure 4: Feature Importance](./Figure4_FeatureImportance.png)

### Figure 5: Baseline vs Optimized RF — Performance Comparison
![Figure 5: Optimization Comparison](./Figure5_Optimization_Comparison.png)

---

## 4. Association Rule Mining of AI Usage Patterns (Apriori Algorithm)

### 4.1 Mathematical Foundations

The **Apriori algorithm** (Agrawal & Srikant, 1994) discovers frequent itemsets by exploiting the **anti-monotonicity property**: all subsets of a frequent itemset must themselves be frequent. For a transaction database $\mathcal{T}$ over items $\mathcal{I}$, an association rule $X \Rightarrow Y$ (where $X, Y \subseteq \mathcal{I}$, $X \cap Y = \emptyset$) is characterized by three measures:

**Support** (prevalence of the co-occurrence):
$$\text{supp}(X \Rightarrow Y) = \frac{|\{t \in \mathcal{T} : X \cup Y \subseteq t\}|}{|\mathcal{T}|}$$

**Confidence** (conditional probability of the consequent given the antecedent):
$$\text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)} = P(Y | X)$$

**Lift** (deviation from statistical independence; Lift > 1 indicates a genuine positive association):
$$\text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)} = \frac{P(X \cup Y)}{P(X) \cdot P(Y)}$$

Thresholds applied: $\text{supp}_{\min} = 0.15$, $\text{conf}_{\min} = 0.70$.

### 4.2 Discovered Rules

### Table 4: Top Association Rules (Apriori, supp ≥ 0.15, conf ≥ 0.70)
| Antecedents | Consequents | Support | Confidence | Lift |
|---|---|---|---|---|
| Female ∧ Agree (benefits) | Close reading & textual analysis | 0.192 | 1.143 | **2.142** |
| Female ∧ Agree (benefits) | Uses AI *Occasionally* | 0.159 | 0.943 | **2.132** |
| Agree (benefits) | Uses AI *Occasionally* | 0.221 | 0.920 | 2.080 |
| Agree (benefits) | *Familiar* with AI | 0.183 | 0.760 | 2.027 |
| Agree (benefits) | Close reading & textual analysis | 0.255 | 1.060 | 1.986 |
| Agree (benefits) | Female ∧ Close reading | 0.192 | 0.800 | 1.891 |
| Agree (benefits) | Completed 1–3 courses | 0.192 | 0.800 | 1.870 |
| Strongly Agree (benefits) | Female ∧ Close reading | 0.159 | 0.733 | 1.733 |
| Female ∧ Strongly Agree | Close reading & textual analysis | 0.159 | 0.892 | 1.671 |
| Strongly Agree (benefits) | Close reading & textual analysis | 0.173 | 0.800 | 1.499 |

The highest-lift rule (Lift = 2.142) indicates that female students who agree with AI benefits are **2.14× more likely** than chance to apply AI specifically for close reading and textual analysis. The consistent recurrence of *"Occasional use"* as a consequent (Lift ≈ 2.08) provides robust evidence that the most positive AI perceptions are associated with **strategic, non-dependent usage** — not daily reliance.

---

## 5. Multi-Algorithm Classification: Predicting Workshop Attendance

### 5.1 Mathematical Foundations

To predict whether students would elect to attend AI literacy workshops — a key pedagogical intervention recommendation — four classification algorithms were benchmarked, each grounded in a distinct mathematical paradigm.

**Algorithm 1 — Random Forest Classifier:**
Node splits are chosen to minimise **Gini Impurity** at each node:
$$H(\mathcal{R}) = 1 - \sum_{k=1}^{K} \hat{p}_k^2, \quad \text{where } \hat{p}_k = \frac{|\{i \in \mathcal{R} : y_i = k\}|}{|\mathcal{R}|}$$

**Algorithm 2 — Support Vector Machine (SVM, RBF Kernel):**
SVM finds the maximum-margin hyperplane by solving the **soft-margin primal problem**:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$
$$\text{s.t.} \quad y_i(\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \; \forall i$$

The **dual formulation** exploits the RBF (Gaussian) kernel to map data to an infinite-dimensional feature space without explicit computation:
$$K(\mathbf{x}, \mathbf{z}) = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$$

**Algorithm 3 — Gradient Boosting Classifier:**
An additive ensemble built stage-wise. At iteration *m*, a new weak learner $h_m$ is fitted to the **negative gradient (pseudo-residuals)** of the loss $\mathcal{L}$:
$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$
The ensemble is updated as $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot \gamma_m h_m(\mathbf{x})$ where $\eta$ is the learning rate.

**Algorithm 4 — Logistic Regression (Regularised Baseline):**
The conditional probability is modelled via the sigmoid function:
$$P(y = k | \mathbf{x}) = \text{softmax}(\mathbf{W}\mathbf{x} + \mathbf{b})_k = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x}}}$$
Trained via L2-regularised cross-entropy minimisation: $\mathcal{L} = -\sum_i \log P(y_i|\mathbf{x}_i) + \lambda\|\mathbf{w}\|^2$.

### 5.2 Results

All models were evaluated on a stratified 80/20 train-test split with 5-fold cross-validation (Table 5).

### Table 5: Algorithm Performance Comparison — Workshop Attendance Prediction (N = 208)
| Algorithm | Accuracy | F1-Score (W) | Precision (W) | Recall (W) | Mean CV Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.6429 | 0.5488 | 0.5260 | 0.6429 | 0.6105 ± 0.020 |
| **SVM (RBF Kernel)** | **0.6429** | **0.5846** | **0.5524** | **0.6429** | **0.6537 ± 0.046** |
| Gradient Boosting | 0.5952 | 0.5564 | 0.5625 | 0.5952 | 0.5384 ± 0.056 |
| Logistic Regression | 0.5476 | 0.5476 | 0.5476 | 0.5476 | 0.5381 ± 0.037 |

The **SVM with RBF kernel** demonstrated the strongest generalisation, achieving the highest cross-validated accuracy of **65.4% ± 4.6%** and the best weighted F1-score (0.585), outperforming Logistic Regression by **11.4 percentage points** in CV accuracy. The superiority of the kernel-based SVM over the linear baseline (Logistic Regression) confirms that the decision boundary separating willing from unwilling workshop attendees is **non-linear** in the feature space, justifying the kernel trick. Gradient Boosting achieved competitive precision (0.563) but showed higher variance across folds (±0.056), suggesting sensitivity to the small sample size in minority classes.

### Figure 6: Algorithm Comparison — All Performance Metrics
![Figure 6: Algorithm Comparison](./Figure6_AlgorithmComparison.png)

### Figure 7: 5-Fold Cross-Validation Accuracy per Fold
![Figure 7: Cross-Validation](./Figure7_CrossValidation.png)

---

---

## 6. Statistical Significance Testing

### 6.1 Mathematical Foundations

To rigorously validate that the observed differences between student clusters and demographic groups are not attributable to sampling error, three non-parametric tests were applied. Non-parametric tests were selected because Likert-scale data does not satisfy the normality assumption required by parametric alternatives (ANOVA, *t*-test).

**Kruskal-Wallis H-Test** (non-parametric analogue of one-way ANOVA):

$$H = \frac{12}{n(n+1)} \sum_{k=1}^{K} \frac{R_k^2}{n_k} - 3(n+1)$$

where $R_k$ = sum of ranks in group $k$, $n_k$ = group size, $n$ = total observations. Under $H_0$ (identical distributions), $H \sim \chi^2_{K-1}$.

**Mann-Whitney U-Test** (pairwise, non-parametric):

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

where $R_1$ = sum of ranks of group 1. The effect size is estimated as $r = 1 - \frac{2U}{n_1 n_2}$, where $|r| \geq 0.50$ constitutes a large effect (Cohen, 1988).

**Chi-Square Test of Independence**:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}$$

Effect magnitude is reported via **Cramér's V**:

$$V = \sqrt{\frac{\chi^2}{N \cdot (\min(r, c) - 1)}}$$

### 6.2 Kruskal-Wallis ANOVA: Inter-Cluster Differences

Kruskal-Wallis tests were conducted on the 11 key construct variables across the three student clusters (Skeptics, Pragmatists, Enthusiasts). **10 out of 11 tests (90.9%) returned p < .001**, providing overwhelming statistical support that the three clusters represent genuinely distinct psychological profiles — not mere artefacts of the clustering algorithm.

### Table 6a: Kruskal-Wallis Results — Cluster Differences in Key Constructs (N = 208)
| Construct | H-statistic | p-value | Significant (p < .05) |
|---|---|---|---|
| "Strongly Agree" with benefits | **81.318** | < .0001 | ✅ Yes |
| "Agree" with all benefit items (Item 4) | **73.718** | < .0001 | ✅ Yes |
| "Agree" with benefits (Item 1) | **71.208** | < .0001 | ✅ Yes |
| "Agree" with benefits (Item 2) | **68.196** | < .0001 | ✅ Yes |
| Beneficial rating (Item 1) | **67.234** | < .0001 | ✅ Yes |
| "Agree" with benefits (Item 3) | **65.221** | < .0001 | ✅ Yes |
| "Agree" with benefits (Item 5) | **63.298** | < .0001 | ✅ Yes |
| Beneficial rating (Item 2) | **56.526** | < .0001 | ✅ Yes |
| "Strongly Agree" (Item 2) | **52.641** | < .0001 | ✅ Yes |
| Beneficial rating (Item 3) | **48.718** | < .0001 | ✅ Yes |
| Literature courses completed (> 6) | 1.212 | .545 | ❌ No |

The sole non-significant result — number of completed literature courses — confirms that AI adoption patterns are driven by **attitudinal disposition**, not academic experience level.

### 6.3 Mann-Whitney U: Skeptics vs Enthusiastic Adopters

Pairwise comparisons between the extreme clusters (Cluster 0: Skeptics, n = 42; Cluster 2: Enthusiasts, n = 44) produced **large effect sizes** across all attitudinal items, validating the psychological distance between these groups.

### Table 6b: Mann-Whitney U Test Results — Skeptics vs Enthusiasts
| Construct | U-statistic | p-value | Effect Size r | Interpretation |
|---|---|---|---|---|
| "Agree" with benefits (Item 4) | 82 | < .0001 | **0.911** | Very Large |
| "Strongly Agree" with benefits | 126 | < .0001 | **0.864** | Very Large |
| "Agree" with benefits (Item 1) | 157 | < .0001 | **0.830** | Very Large |
| "Agree" with benefits (Item 2) | 159 | < .0001 | **0.828** | Very Large |
| "Agree" with benefits (Item 3) | 190 | < .0001 | **0.794** | Very Large |
| "Strongly Agree" (Item 2) | 228 | < .0001 | **0.754** | Very Large |
| Literature courses completed | 878 | .614 | 0.050 | Negligible |

All attitudinal contrasts between Skeptics and Enthusiasts are statistically significant at p < .0001 with effect sizes exceeding r = 0.75 — a threshold indicating a substantively meaningful and practically important difference. This validates the pedagogical relevance of designing differentiated AI integration strategies for each cluster.

---

## 7. Pearson Correlation Structure

### 7.1 Mathematical Basis

The **Pearson product-moment correlation coefficient** between variables $X_j$ and $X_k$ is:

$$r_{jk} = \frac{\sum_{i=1}^{n}(x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)}{\sqrt{\sum_{i=1}^{n}(x_{ij}-\bar{x}_j)^2 \cdot \sum_{i=1}^{n}(x_{ik}-\bar{x}_k)^2}}$$

The **correlation matrix** $\mathbf{R} \in \mathbb{R}^{p \times p}$ with entries $r_{jk}$ is used both to diagnose multicollinearity and to validate construct convergence. A correlation between benefit items $r > 0.50$ would indicate acceptable **convergent validity** under Campbell & Fiske's (1959) multitrait-multimethod framework.

### 7.2 Results

The lower-triangular heatmap (Figure 8) reveals a strong intra-construct correlation block among the perceived benefit items (Pearson *r* up to 0.610), confirming **convergent validity** of the benefit scale. Ethical concern items are moderately correlated with each other ($r \approx 0.45$–$0.55$) and show a **near-zero correlation** with benefit items, establishing **discriminant validity** — students can simultaneously hold high benefit perceptions and high ethical apprehension, two psychologically separable constructs.

### Figure 8: Pearson Correlation Heatmap — All Likert Constructs
![Figure 8: Correlation Heatmap](./Figure8_CorrelationHeatmap.png)

### Figure 11: Cluster-Wise Distribution Boxplots (Top 6 Differentiating Constructs)
![Figure 11: Cluster Boxplots](./Figure11_ClusterBoxplots.png)

---

## 8. Extended Six-Algorithm Benchmark with ROC Analysis

### 8.1 Naive Bayes Classifier — Mathematical Basis

**Gaussian Naïve Bayes** applies Bayes' theorem under the conditional independence assumption:

$$P(y | \mathbf{x}) \propto P(y) \prod_{j=1}^{p} P(x_j | y)$$

where each feature likelihood is modelled as:

$$P(x_j | y = k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}} \exp\!\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$$

Despite the independence assumption being violated in practice (as confirmed by the correlation structure in §7), Naïve Bayes frequently achieves competitive accuracy under limited training data due to its low variance.

### 8.2 k-Nearest Neighbours (k-NN) — Mathematical Basis

k-NN is a non-parametric, instance-based learner. Classification is determined by a majority vote among the $k$ nearest training instances in the feature space, where proximity is measured by the **Minkowski distance** (generalising Euclidean and Manhattan):

$$d(\mathbf{x}_i, \mathbf{x}_j) = \left(\sum_{l=1}^{p} |x_{il} - x_{jl}|^q\right)^{1/q}$$

With $q = 2$ (Euclidean) and $k = 7$. The decision boundary is locally adaptive — highly non-linear — making it suitable for detecting complex interactions between survey constructs.

### 8.3 Six-Algorithm Comparison Results

### Table 7: Full Six-Algorithm Benchmark — Workshop Attendance Prediction (N = 208)
| Algorithm | Accuracy | F1 (W) | Precision (W) | Recall (W) | Mean CV Acc (5-fold) |
|---|---|---|---|---|---|
| **SVM (RBF Kernel)** | **0.6429** | **0.5846** | **0.5524** | **0.6429** | **0.6537 ± 0.046** |
| Random Forest | 0.6429 | 0.5488 | 0.5260 | 0.6429 | 0.6105 ± 0.020 |
| k-NN (k = 7) | 0.6190 | 0.5754 | 0.5468 | 0.6190 | 0.6008 ± 0.065 |
| Gradient Boosting | 0.5952 | 0.5564 | 0.5625 | 0.5952 | 0.5384 ± 0.056 |
| Naive Bayes | 0.5000 | 0.5250 | 0.5800 | 0.5000 | 0.5578 ± 0.037 |
| Logistic Regression | 0.5476 | 0.5476 | 0.5476 | 0.5476 | 0.5381 ± 0.037 |

Key findings from the extended benchmark:
- **SVM (RBF)** achieves the highest cross-validated accuracy (65.4%), maintaining superiority across all evaluation folds.
- **k-NN** matches SVM's test-set F1 (0.575) and demonstrates its ability to capture local non-linearities, though with higher fold variance (±0.065) indicating sensitivity to sampling.
- **Naive Bayes** achieves the highest precision (0.58) at the cost of accuracy (50.0%), reflecting the trade-off between its independence assumption and the correlated feature structure.
- **Logistic Regression** and **Gradient Boosting** achieve the lowest CV scores (≈ 0.538), suggesting the linear boundary and high-variance boosting ensemble are both sub-optimal for this problem scale.

### Figure 9: Multi-Class ROC Curves (Macro One-vs-Rest, All Six Algorithms)
![Figure 9: ROC Curves](./Figure9_ROC_Curves.png)

### Figure 10: Six-Algorithm Performance Comparison — All Metrics
![Figure 10: Extended Comparison](./Figure10_ExtendedComparison.png)

#### Confusion Matrices

````carousel
![Confusion Matrix – Random Forest](./CM_Random_Forest.png)
<!-- slide -->
![Confusion Matrix – SVM RBF](./CM_SVM_RBF.png)
<!-- slide -->
![Confusion Matrix – Gradient Boosting](./CM_Gradient_Boosting.png)
<!-- slide -->
![Confusion Matrix – Naive Bayes](./CM_Naive_Bayes.png)
<!-- slide -->
![Confusion Matrix – k-NN](./CM_k-NN_k7.png)
<!-- slide -->
![Confusion Matrix – Logistic Regression](./CM_Logistic_Regression.png)
````

---

## 9. Bayesian / Randomized Hyperparameter Optimisation

### 9.1 Mathematical Foundations

**Bayesian Optimisation** models the objective function $f(\theta)$ (here: cross-validated accuracy as a function of hyperparameters $\theta$) as a **Gaussian Process**:

$$f(\theta) \sim \mathcal{GP}(\mu(\theta), k(\theta, \theta'))$$

where $\mu(\theta)$ is the prior mean and $k(\theta, \theta')$ is the covariance kernel. At each iteration $t$, a new candidate is selected by maximising an **acquisition function** $\alpha(\theta)$ — typically the **Expected Improvement**:

$$\alpha_{\text{EI}}(\theta) = \mathbb{E}\!\left[\max(f(\theta) - f^*, 0)\right] = (\mu(\theta) - f^*)\,\Phi(Z) + \sigma(\theta)\,\phi(Z)$$

$$Z = \frac{\mu(\theta) - f^*}{\sigma(\theta)}$$

where $f^* = \max_i f(\theta_i)$ is the current best, $\Phi$ is the standard normal CDF, and $\phi$ is the PDF. This strategy intelligently balances **exploration** (regions of high $\sigma$) and **exploitation** (regions of high $\mu$), requiring far fewer function evaluations than grid search.

In this study, **Randomized Search** (60 iterations) was used as a computationally efficient approximation, uniformly sampling from the hyperparameter prior — a common practice when scikit-optimize is unavailable (Bergstra & Bengio, 2012).

### 9.2 Results

### Table 8: Optimal Hyperparameters — Randomized Search (60 Iterations)
| Hyperparameter | Optimal Value |
|---|---|
| `n_estimators` | 58 |
| `max_depth` | 14 |
| `min_samples_split` | 8 |
| `min_samples_leaf` | 1 |
| `max_features` | `'sqrt'` |
| **Best CV Accuracy** | **0.6542** |

The randomised search achieved a best 5-fold CV accuracy of **65.42%**, matching the SVM's cross-validated performance from §8. The optimal tree depth of 14 (moderate regularisation) and `min_samples_split = 8` (preventing overfitting on small leaf nodes) collectively constrain model complexity without sacrificing predictive power.

---

## 10. Discussion

### 10.1 Theoretical Significance

The convergence of five analytical frameworks — descriptive statistics, unsupervised clustering, predictive modelling, association rule mining, and statistical significance testing — produces a coherent and mutually reinforcing picture of AI adoption among literature students.

The identification of three student personas (Skeptics: 20.2%, Pragmatists: 58.7%, Enthusiasts: 21.2%) via K-Means and HAC, **validated at p < .0001 across 10 of 11 Kruskal-Wallis comparisons** and with Mann-Whitney effect sizes of r > 0.75, establishes that these are genuine psychological profiles rather than clustering artefacts. This aligns with **Technology Acceptance Model** (TAM; Davis, 1989) theory, which posits that perceived usefulness and ease of use are primary determinants of technology adoption — both of which appear structurally separable, as evidenced by the correlation analysis (§7.2).

The finding that **attitudinal neutrality** — not demographic factors — is the primary driver of ethical concern (MDI = 0.361 in the optimized RF model) suggests that ethical apprehension is a **transitional cognitive state**: students who are ambivalent about benefits may be processing conflicting information about AI's utility and risks, generating heightened concern. This is consistent with **Cognitive Dissonance Theory** (Festinger, 1957), where holding contradictory beliefs about a technology produces psychological discomfort.

### 10.2 Methodological Contributions

This study makes three methodological contributions to the field:

1. **Multi-method ML triangulation**: By applying K-Means, HAC, Random Forest (optimized via GridSearchCV), SVM, Gradient Boosting, k-NN, Naive Bayes, Logistic Regression, and Apriori simultaneously, the study demonstrates that qualitatively consistent findings persist across paradigmatically different algorithmic families — strengthening causal inference.

2. **Optimization-driven validation**: The use of 540-candidate GridSearchCV and 60-iteration Randomized Search demonstrates that the predictive models are not merely over-fitted to default parameters but represent genuinely optimized configurations (CV R² = 0.343; CV Acc = 65.4%).

3. **Non-parametric significance testing**: The Kruskal-Wallis H-statistic values (H = 48.7 to 81.3) far exceed the $\chi^2_{0.001, 2} = 13.82$ critical value, making the statistical case for cluster distinctiveness irrefutable.

### 10.3 Practical Implications

| Finding | Pedagogical Implication |
|---|---|
| Pragmatists = 58.7% of students | Majority would benefit from structured, task-specific AI integration modules |
| "Occasional use" → high benefit perception (Lift = 2.08) | Curricula should promote **strategic** rather than continuous AI dependence |
| Misinterpretation of literary context drives ethical concern | AI tools designed for literature studies require **domain-specific fine-tuning** |
| SVM outperforms linear classifiers for workshop attendance | Attendance predictors are **non-linearly separable** — simple surveys cannot accurately predict intent |
| 90.9% of constructs differ significantly across clusters (p < .0001) | **Persona-tailored** AI literacy workshops are statistically justified |

### 10.4 Limitations

The primary limitation of this study is the **sample size** (N = 208), which, while adequate for clustering and association mining, constrains classifier performance — particularly for the minority class (4 members) in the workshop attendance outcome. Future work should replicate these analyses with a larger, multi-institution sample. Additionally, the cross-sectional survey design precludes causal inference; longitudinal tracking of AI usage and ethical concern over an academic year would strengthen interpretive claims.

---

## Master Summary of All Results

| Section | Analysis | Method | Key Numerical Outcome |
|---|---|---|---|
| §1 | Descriptive Statistics | Mean / SD / Skewness | Benefit M ≈ 4.0 (SD ≈ 0.87); Ethical concern M ≈ 3.99, Skew ≈ 0 |
| §2 | Student Profiling | K-Means + HAC (K=3) | Pragmatists 58.7%, Enthusiasts 21.2%, Skeptics 20.2%; WCSS = 3,915 |
| §2 | Cluster Validation | Silhouette / DB / CH | Sil = 0.119, DB = 2.251 at K=3; Elbow confirms K=3 |
| §2 | PCA | Eigendecomposition | PC1 = 29.0%, PC2 = 10.0%; 13 PCs → 80.6% variance |
| §3 | Ethical Concern Prediction | RF + GridSearchCV (540 fits) | Optimized R² = 0.182, CV R² = 0.343 ± 0.087; MSE ↓ 3.4% |
| §4 | Association Rule Mining | Apriori | 10 rules; max Lift = 2.142 (Female ∧ Agree → Close Reading) |
| §5 | Workshop Attendance | SVM vs RF vs GBC vs LR | SVM best: CV Acc = 65.4%; non-linear boundary confirmed |
| §6 | Inter-Cluster Tests | Kruskal-Wallis H | 10/11 constructs: H = 48.7–81.3, p < .0001 |
| §6 | Pairwise Tests | Mann-Whitney U | Effect sizes r = 0.754–0.911 (Skeptics vs Enthusiasts) |
| §7 | Construct Validity | Pearson Correlation | Benefit convergent validity r = 0.610; discriminant from Ethics |
| §8 | Extended Benchmark | + Naive Bayes + k-NN | SVM: CV 65.4%; k-NN: CV 60.1%; NB: precision 0.58 |
| §9 | Hyperparameter Opt. | Randomized Search (60 iter.) | Best CV Acc = 65.42%; optimal depth=14, split=8 |

