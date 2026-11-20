"""
Idea 1: Predicting Student Engagement and Academic Impact
=========================================================
Target   : Perceived academic impact of AI (col 32: '4- Moderate impact')
           Likert 1-5 → binarised: 4-5 = "High Impact", 1-3 = "Low Impact"
Features : Demographics (cols 1-3), AI familiarity (col 6), frequency (col 9),
           tools used (cols 7-8), task types (cols 10-12), course exposure (col 3)
Models   : Random Forest, SVM (RBF), Gradient Boosting, k-NN, Naive Bayes, Logistic Regression
Outputs  : All tables (CSV) and figures (PNG) saved to this first/ folder.
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from scipy import stats

# ── Paths ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "..", "XLS data.csv")

def clean_likert(val):
    if pd.isna(val): return np.nan
    m = re.search(r'^\d', str(val).strip())
    return float(m.group()) if m else np.nan

def save(fig, name):
    p = os.path.join(SCRIPT_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

# ── 1. Load ─────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH, header=1)
print(f"  {len(df)} rows × {len(df.columns)} cols")

# ── 2. Define Target and Features by column index ──────
# Col 32: "4- Moderate impact" — Engagement/Academic Impact (Likert 1-5)
TARGET_IDX   = 32
FEATURE_IDXS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

target_col   = df.columns[TARGET_IDX]
feature_cols = [df.columns[i] for i in FEATURE_IDXS]

print(f"  Target  : col {TARGET_IDX} = '{target_col}'")
print(f"  Features: {[df.columns[i] for i in FEATURE_IDXS]}")

# ── 3. Build target ─────────────────────────────────────
df["_raw"] = df[target_col].apply(clean_likert)
df = df.dropna(subset=["_raw"])
df["Engagement"] = df["_raw"].apply(lambda x: "High Impact" if x >= 4 else "Low Impact")
print(f"\n  Target distribution:\n{df['Engagement'].value_counts().to_string()}")

# ── 4. Build features ────────────────────────────────────
X = df[feature_cols].fillna("Unknown").astype(str).copy()
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

le_y = LabelEncoder()
y    = le_y.fit_transform(df["Engagement"])
classes = le_y.classes_
print(f"  Classes: {classes}  (0={classes[0]}, 1={classes[1]})")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

# ── 5. Train all six models ──────────────────────────────
MODELS = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":           Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=42))]),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "k-NN (k=7)":          Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes":         Pipeline([("sc", StandardScaler()), ("m", GaussianNB())]),
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
}

cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
PAL  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]

fig_roc, ax_roc = plt.subplots(figsize=(9, 7))

for idx, (name, model) in enumerate(MODELS.items()):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    cv_s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # ROC
    try:
        proba = model.predict_proba(X_te)
        fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        roc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=PAL[idx], lw=2,
                    label=f"{name} (AUC = {roc_val:.3f})")
    except Exception:
        roc_val = float("nan")

    rows.append({"Algorithm": name, "Accuracy": round(acc,4),
                 "F1 (W)": round(f1,4), "Precision": round(prec,4),
                 "Recall": round(rec,4), "ROC-AUC": round(roc_val,4),
                 "CV Mean Acc": round(cv_s.mean(),4), "CV Std": round(cv_s.std(),4)})

    print(f"\n[{name}]  Acc={acc:.4f}  F1={f1:.4f}  CV={cv_s.mean():.4f}±{cv_s.std():.4f}")
    print(classification_report(y_te, y_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    cm_v = confusion_matrix(y_te, y_pred)
    fig_c, ax_c = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_v, annot=True, fmt="d", cmap="Blues", ax=ax_c,
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax_c.set_xlabel("Predicted"); ax_c.set_ylabel("True")
    ax_c.set_title(f"Confusion Matrix – {name}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    save(fig_c, f"CM_{name.replace(' ','_').replace('(','').replace(')','')}.png")

# ── 6. ROC plot ──────────────────────────────────────────
ax_roc.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random Chance")
ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Predicting High vs Low Academic Impact of AI (N=208)")
ax_roc.legend(fontsize=9, loc="lower right"); ax_roc.grid(True, alpha=0.3)
save(fig_roc, "Figure_ROC.png")

# ── 7. Algorithm comparison bar chart ────────────────────
res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(SCRIPT_DIR, "Table_Algorithms.csv"), index=False)
print("\nSaved Table_Algorithms.csv")

fig_b, ax_b = plt.subplots(figsize=(13, 6))
metrics_plot = ["Accuracy", "F1 (W)", "Precision", "Recall"]
x = np.arange(len(rows)); w = 0.18
bar_c = ["#4C72B0","#DD8452","#55A868","#C44E52"]
for i, m in enumerate(metrics_plot):
    ax_b.bar(x+(i-1.5)*w, res_df[m], w, label=m, color=bar_c[i], alpha=0.87)
ax_b.set_xticks(x)
ax_b.set_xticklabels(res_df["Algorithm"], fontsize=9, rotation=10)
ax_b.set_ylim(0, 1.2); ax_b.set_ylabel("Score", fontsize=11)
ax_b.set_title("Six-Algorithm Benchmark: Predicting Student Engagement / Academic Impact",
               fontsize=12, fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.axhline(0.7, color="grey", linestyle="--", lw=0.8, alpha=0.6, label="0.70 line")
save(fig_b, "Figure_AlgorithmComparison.png")

# ── 8. CV accuracy per fold ──────────────────────────────
fig_cv, ax_cv = plt.subplots(figsize=(10, 5))
for idx, (name, model) in enumerate(MODELS.items()):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    ax_cv.plot(range(1, 6), scores, marker="o", label=name, color=PAL[idx], lw=2)
ax_cv.set(xlabel="Fold", ylabel="Accuracy",
          title="5-Fold CV Accuracy per Fold — Student Engagement Prediction")
ax_cv.set_xticks(range(1,6)); ax_cv.legend(fontsize=9); ax_cv.grid(True, alpha=0.3)
save(fig_cv, "Figure_CV_Folds.png")

# ── 9. Feature importance (Random Forest) ────────────────
rf_model = MODELS["Random Forest"]
imp = rf_model.feature_importances_
order = np.argsort(imp)[::-1]
n_top = len(feature_cols)
labels = [str(X.columns[i])[:50] for i in order[::-1]]
vals   = imp[order[::-1]]

fig_fi, ax_fi = plt.subplots(figsize=(11, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_top))
ax_fi.barh(range(n_top), vals, color=colors, edgecolor="black", linewidth=0.4)
ax_fi.set_yticks(range(n_top)); ax_fi.set_yticklabels(labels, fontsize=9)
ax_fi.set_xlabel("Mean Decrease in Impurity (Gini)", fontsize=11)
ax_fi.set_title("Feature Importance — Predicting Academic Impact (Random Forest)",
                fontsize=12, fontweight="bold")
ax_fi.axvline(np.mean(imp), color="navy", linestyle="--", lw=1.2,
              label=f"Mean importance = {np.mean(imp):.4f}")
ax_fi.legend(fontsize=9)
save(fig_fi, "Figure_FeatureImportance.png")

fi_df = pd.DataFrame({"Feature": [str(X.columns[i]) for i in order],
                       "Importance": np.round(imp[order], 6)})
fi_df.to_csv(os.path.join(SCRIPT_DIR, "Table_FeatureImportance.csv"), index=False)
print("Saved Table_FeatureImportance.csv")

# ── 10. PCA scatter: true vs predicted ──────────────────
svm_model = MODELS["SVM (RBF)"]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
y_pred_all = svm_model.predict(X)
pc1_var = pca.explained_variance_ratio_[0]
pc2_var = pca.explained_variance_ratio_[1]

fig_pca, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, labels_arr, title_suffix in [
    (axes[0], y, "True Labels"),
    (axes[1], y_pred_all, "SVM Predicted Labels")
]:
    for ci, (cls, col) in enumerate(zip(classes, ["#C44E52","#4C72B0"])):
        mask = labels_arr == ci
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=col,
                   label=f"{cls}", alpha=0.65, s=60, edgecolors="white", lw=0.3)
    ax.set(xlabel=f"PC1 ({pc1_var:.1%} var)", ylabel=f"PC2 ({pc2_var:.1%} var)",
           title=f"PCA Projection — {title_suffix}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
save(fig_pca, "Figure_PCA.png")

# ── 11. Statistical significance: High vs Low groups ─────
print("\n=== Mann-Whitney U: High vs Low Engagement Groups ===")
df_mw = X.copy(); df_mw["Target"] = y
mw_rows = []
for col in X.columns:
    g0 = df_mw.loc[df_mw["Target"]==0, col].values
    g1 = df_mw.loc[df_mw["Target"]==1, col].values
    if len(g0) < 2 or len(g1) < 2: continue
    if np.all(g0 == g0[0]) and np.all(g1 == g1[0]): continue
    try:
        u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        r = 1 - (2*u) / (len(g0)*len(g1))
        mw_rows.append({"Feature": str(col)[:60], "U": round(u,2),
                        "p-value": round(p,4), "Effect r": round(abs(r),4),
                        "Significant": "Yes" if p < 0.05 else "No"})
        print(f"  {str(col)[:48]:50s} U={u:.0f}  p={p:.4f}  r={r:.3f} {'*' if p<0.05 else ''}")
    except Exception:
        continue
mw_df = pd.DataFrame(mw_rows).sort_values("Effect r", ascending=False)
mw_df.to_csv(os.path.join(SCRIPT_DIR, "Table_MannWhitney.csv"), index=False)
print("Saved Table_MannWhitney.csv")

# ── 12. Likert distribution by Engagement group ──────────
fig_d, ax_d = plt.subplots(figsize=(10, 5))
df_plot = df[["_raw", "Engagement"]].copy()
for cls, col in zip(classes, ["#C44E52","#4C72B0"]):
    subset = df_plot.loc[df_plot["Engagement"]==cls, "_raw"]
    ax_d.hist(subset, bins=[0.5,1.5,2.5,3.5,4.5,5.5], alpha=0.7,
              color=col, edgecolor="black", label=cls, density=True, rwidth=0.8)
ax_d.set(xlabel="Likert Score (1=Low, 5=High)", ylabel="Proportion",
         title="Distribution of Academic Impact Ratings by Engagement Group",
         xticks=[1,2,3,4,5])
ax_d.legend(fontsize=10)
save(fig_d, "Figure_Distribution.png")

# ── Summary ──────────────────────────────────────────────
best = max(rows, key=lambda r: r["CV Mean Acc"])
print(f"\n{'='*55}")
print(f"IDEA 1 COMPLETE — Predicting Student Academic Engagement")
print(f"{'='*55}")
print(f"  Target   : '{target_col}' (cols {TARGET_IDX})")
print(f"  N        : {len(df)} students  |  Features: {len(feature_cols)}")
print(f"  Best CV  : {best['Algorithm']}  →  {best['CV Mean Acc']:.4f} ± {best['CV Std']:.4f}")
print(f"  Figures  : ROC, Comparison, CV Folds, Feature Importance, PCA, Distribution")
print(f"  Tables   : Algorithms, FeatureImportance, MannWhitney")
