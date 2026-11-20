"""
advanced_analysis.py
Extends the analysis with:
  1. Statistical significance testing (Chi-Square, Kruskal-Wallis, Mann-Whitney U)
  2. Naive Bayes & k-NN algorithms
  3. ROC curves (one-vs-rest, multi-class)
  4. Confusion matrices for all classifiers
  5. Bayesian Optimisation for RF hyperparameters
  6. Correlation heatmap of Likert constructs
  7. Cluster-wise ANOVA of key engagement variables
"""

import pandas as pd
import numpy as np
import os, re, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              roc_auc_score, roc_curve, auc,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clean_likert(val):
    if pd.isna(val):
        return np.nan
    m = re.search(r'^\d', str(val).strip())
    return float(m.group()) if m else np.nan


def load_data(script_dir):
    data_path = os.path.join(script_dir, "..", "data", "survey_anonymised.csv")
    df = pd.read_csv(data_path, header=0)
    return df


def encode_features(df, target_col):
    feature_cols = [c for c in df.columns
                    if c != target_col and "timestamp" not in str(c).lower()]
    X = df[feature_cols].fillna("Unknown").astype(str).copy()
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    le_y = LabelEncoder()
    y = le_y.fit_transform(df[target_col].astype(str))
    return X, y, le_y.classes_


def get_likert_matrix(df):
    likert_cols = []
    for col in df.columns:
        nn = df[col].dropna().astype(str)
        if len(nn) and nn.str.match(r'^\d').mean() > 0.2:
            likert_cols.append(col)
    dfn = pd.DataFrame({c: df[c].apply(clean_likert) for c in likert_cols})
    for c in dfn.columns:
        dfn[c] = dfn[c].fillna(dfn[c].median())
    dfn = dfn.dropna(axis=1, how='all')
    return dfn


# ─────────────────────────────────────────────
# 1. Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(dfn, script_dir):
    corr = dfn.corr()
    short_cols = [str(c)[:30] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=False, linewidths=0.3, ax=ax,
                cbar_kws={'label': "Pearson r", 'shrink': 0.7})
    ax.set_xticklabels(short_cols, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(short_cols, rotation=0, fontsize=6)
    ax.set_title('Pearson Correlation Heatmap – Likert Constructs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(script_dir, "Figure8_CorrelationHeatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")

    # Extract strongest correlations
    corr_vals = (corr.where(~mask).stack()
                     .reset_index()
                     .rename(columns={'level_0': 'Var1', 'level_1': 'Var2', 0: 'r'})
                     .reindex(columns=['Var1', 'Var2', 'r']))
    corr_vals['abs_r'] = corr_vals['r'].abs()
    top_corr = corr_vals.nlargest(15, 'abs_r')[['Var1', 'Var2', 'r']].round(4)
    top_corr.to_csv(os.path.join(script_dir, "Table9_TopCorrelations.csv"), index=False)
    print("Saved Table9_TopCorrelations.csv")
    return top_corr


# ─────────────────────────────────────────────
# 2. Statistical Significance Tests
# ─────────────────────────────────────────────
def run_significance_tests(df, dfn, script_dir):
    print("\n\n=== STATISTICAL SIGNIFICANCE TESTS ===")

    # --- Kruskal-Wallis ANOVA across clusters ---
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(dfn)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km.fit_predict(X_sc)
    dfn_c = dfn.copy()
    dfn_c['Cluster'] = clusters

    kw_results = []
    for col in dfn.columns[:12]:          # test first 12 constructs
        groups = [dfn_c.loc[dfn_c['Cluster'] == k, col].values for k in range(3)]
        # Skip if any group is constant or empty
        if any(len(g) == 0 or np.all(g == g[0]) for g in groups):
            continue
        try:
            stat, p = stats.kruskal(*groups)
        except ValueError:
            continue
        kw_results.append({'Variable': str(col)[:60], 'H-statistic': round(stat, 4), 'p-value': round(p, 4),
                            'Significant (p<.05)': 'Yes' if p < 0.05 else 'No'})
        print(f"  KW: {str(col)[:50]:52s} H={stat:.3f} p={p:.4f} {'*' if p < 0.05 else ''}")

    kw_df = pd.DataFrame(kw_results)
    kw_df.to_csv(os.path.join(script_dir, "Table10_KruskalWallis.csv"), index=False)
    print("Saved Table10_KruskalWallis.csv")

    # --- Chi-Square: Gender × AI Usage Frequency ---
    # Find relevant columns
    gender_col = freq_col = None
    for col in df.columns:
        if 'gender' in str(col).lower() or 'male' in str(col).lower():
            gender_col = col
        if 'how often' in str(col).lower() or 'frequency' in str(col).lower():
            freq_col = col
    if gender_col and freq_col:
        ct = pd.crosstab(df[gender_col].astype(str), df[freq_col].astype(str))
        chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
        print(f"\n  Chi-Square (Gender × Usage Frequency):")
        print(f"    χ²({dof}) = {chi2:.3f}, p = {p_chi:.4f}, N = {ct.values.sum()}")
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        print(f"    Cramér's V = {v:.4f}")
        chi_result = pd.DataFrame([{'Test': "Chi-Square (Gender × Usage Freq)",
                                    'chi2': round(chi2, 4), 'df': dof,
                                    'p-value': round(p_chi, 4), "Cramér's V": round(v, 4)}])
        chi_result.to_csv(os.path.join(script_dir, "Table11_ChiSquare.csv"), index=False)
        print("Saved Table11_ChiSquare.csv")
    else:
        print("  Chi-Square: required columns not found.")
        chi_result = pd.DataFrame()

    # --- Mann-Whitney U: Cluster 0 (Skeptics) vs Cluster 2 (Enthusiasts) ---
    mw_results = []
    for col in dfn.columns[:8]:
        g0 = dfn_c.loc[dfn_c['Cluster'] == 0, col].values
        g2 = dfn_c.loc[dfn_c['Cluster'] == 2, col].values
        stat_u, p_u = stats.mannwhitneyu(g0, g2, alternative='two-sided')
        r_effect = 1 - (2 * stat_u) / (len(g0) * len(g2))
        mw_results.append({'Variable': str(col)[:60], 'U': round(stat_u, 2),
                            'p-value': round(p_u, 4), 'r (effect)': round(r_effect, 4),
                            'Significant': 'Yes' if p_u < 0.05 else 'No'})
        print(f"  MWU (C0 vs C2): {str(col)[:45]:47s} U={stat_u:.0f} p={p_u:.4f} r={r_effect:.3f}")

    mw_df = pd.DataFrame(mw_results)
    mw_df.to_csv(os.path.join(script_dir, "Table12_MannWhitney.csv"), index=False)
    print("Saved Table12_MannWhitney.csv")
    return kw_df, chi_result, mw_df


# ─────────────────────────────────────────────
# 3. Extended Multi-Algorithm Classification
#    + ROC Curves + Confusion Matrices
# ─────────────────────────────────────────────
def run_extended_classifiers(df, script_dir):
    print("\n\n=== EXTENDED CLASSIFIER BENCHMARK ===")

    # Find target
    target_col = None
    for col in df.columns:
        if "attending workshops or training sessions" in str(col).lower():
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
    df = df.dropna(subset=[target_col])

    X, y, classes = encode_features(df, target_col)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_bin = label_binarize(y_te, classes=np.unique(y))

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":          Pipeline([('sc', StandardScaler()), ('m', SVC(kernel='rbf', probability=True, random_state=42))]),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "Naive Bayes":        Pipeline([('sc', StandardScaler()), ('m', GaussianNB())]),
        "k-NN (k=7)":         Pipeline([('sc', StandardScaler()), ('m', KNeighborsClassifier(n_neighbors=7))]),
        "Logistic Regression":Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000, random_state=42))]),
    }

    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ROC figure
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
    n_classes = len(np.unique(y))

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        cv_s = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_cv = cv_s.mean()
        std_cv  = cv_s.std()

        # ROC (macro OvR)
        try:
            proba = model.predict_proba(X_te)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
                roc_auc_val = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color=palette[idx], lw=2,
                            label=f'{name} (AUC={roc_auc_val:.3f})')
            else:
                # Macro-average OvR
                fpr_all, tpr_all = {}, {}
                roc_auc_val = 0
                for ci in range(n_classes):
                    if ci < y_bin.shape[1]:
                        fpr_c, tpr_c, _ = roc_curve(y_bin[:, ci], proba[:, ci])
                        roc_auc_val += auc(fpr_c, tpr_c)
                        fpr_all[ci] = fpr_c; tpr_all[ci] = tpr_c
                roc_auc_val /= n_classes
                all_fpr = np.unique(np.concatenate(list(fpr_all.values())))
                mean_tpr = np.zeros_like(all_fpr)
                for ci in range(n_classes):
                    if ci < y_bin.shape[1]:
                        mean_tpr += np.interp(all_fpr, fpr_all[ci], tpr_all[ci])
                mean_tpr /= n_classes
                ax_roc.plot(all_fpr, mean_tpr, color=palette[idx], lw=2,
                            label=f'{name} (macro AUC={roc_auc_val:.3f})')
        except Exception:
            roc_auc_val = float('nan')

        rows.append({'Algorithm': name, 'Accuracy': round(acc, 4),
                     'F1 (W)': round(f1, 4), 'Precision (W)': round(prec, 4),
                     'Recall (W)': round(rec, 4), 'ROC-AUC': round(roc_auc_val, 4),
                     'CV Mean Acc': round(mean_cv, 4), 'CV Std': round(std_cv, 4)})

        print(f"\n[{name}]  Acc={acc:.4f}  F1={f1:.4f}  CV={mean_cv:.4f}±{std_cv:.4f}")
        print(classification_report(y_te, y_pred, zero_division=0))

        # Confusion Matrix — draw with seaborn to avoid matplotlib label-count issue
        cm_val = confusion_matrix(y_te, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        unique_preds = np.unique(np.concatenate([y_te, y_pred]))
        tick_labels  = [str(classes[i]) if i < len(classes) else str(i) for i in unique_preds]
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=tick_labels, yticklabels=tick_labels, cbar=False)
        ax_cm.set_xlabel('Predicted Label', fontsize=10)
        ax_cm.set_ylabel('True Label', fontsize=10)
        ax_cm.set_title(f'Confusion Matrix – {name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        cm_path = os.path.join(script_dir, f"CM_{safe_name}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()

    # Finalise ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Chance')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves — Multi-Algorithm Comparison (Macro OvR)', fontsize=13, fontweight='bold')
    ax_roc.legend(fontsize=9, loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(script_dir, "Figure9_ROC_Curves.png")
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)
    print(f"\nSaved {roc_path}")

    # Summary table
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(script_dir, "Table13_ExtendedComparison.csv"), index=False)
    print("Saved Table13_ExtendedComparison.csv")

    # Grouped bar chart (extended)
    fig_bar, ax_bar = plt.subplots(figsize=(14, 6))
    metric_cols = ['Accuracy', 'F1 (W)', 'Precision (W)', 'Recall (W)']
    x = np.arange(len(rows))
    bar_w = 0.18
    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for mi, mc in enumerate(metric_cols):
        vals = [r[mc] for r in rows]
        ax_bar.bar(x + (mi - 1.5)*bar_w, vals, bar_w, label=mc, color=bar_colors[mi], alpha=0.87)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([r['Algorithm'] for r in rows], fontsize=8, rotation=12)
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_ylabel('Score', fontsize=11)
    ax_bar.set_title('Six-Algorithm Benchmark: Workshop Attendance Prediction', fontsize=13, fontweight='bold')
    ax_bar.axhline(0.65, color='grey', linestyle='--', linewidth=0.8, alpha=0.6, label='65% threshold')
    ax_bar.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure10_ExtendedComparison.png"), dpi=300)
    plt.close()
    print("Saved Figure10_ExtendedComparison.png")

    return res_df


# ─────────────────────────────────────────────
# 4. Bayesian Optimisation (scikit-optimize)
# ─────────────────────────────────────────────
def run_bayesian_optimisation(df, script_dir):
    print("\n\n=== BAYESIAN OPTIMISATION ===")
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer
        HAS_SKOPT = True
    except ImportError:
        HAS_SKOPT = False
        print("  scikit-optimize not installed — running manual random search instead.")

    target_col = None
    for col in df.columns:
        if "attending workshops or training sessions" in str(col).lower():
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
    df = df.dropna(subset=[target_col])
    X, y, classes = encode_features(df, target_col)

    if HAS_SKOPT:
        search_spaces = {
            'n_estimators': Integer(50, 500),
            'max_depth':    Integer(2, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf':  Integer(1, 10),
            'max_features': Real(0.1, 1.0),
        }
        bayes_opt = BayesSearchCV(
            RandomForestClassifier(random_state=42),
            search_spaces, n_iter=40, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0, refit=True)
        bayes_opt.fit(X, y)
        best_p  = bayes_opt.best_params_
        best_sc = bayes_opt.best_score_
        print(f"  Best params  : {best_p}")
        print(f"  Best CV Acc  : {best_sc:.4f}")
        bo_df = pd.DataFrame([{'Method': 'Bayesian Optimisation (BayesSearchCV)',
                                **{k: v for k, v in best_p.items()},
                                'Best CV Accuracy': round(best_sc, 4)}])
    else:
        # Fallback: RandomizedSearchCV with scipy distributions
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
            'n_estimators': stats.randint(50, 500),
            'max_depth':    [None] + list(range(3, 20)),
            'min_samples_split': stats.randint(2, 20),
            'min_samples_leaf':  stats.randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
        }
        rs = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_dist, n_iter=60, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0, refit=True)
        rs.fit(X, y)
        best_p  = rs.best_params_
        best_sc = rs.best_score_
        print(f"  Best params (RandomSearch): {best_p}")
        print(f"  Best CV Acc              : {best_sc:.4f}")
        bo_df = pd.DataFrame([{'Method': 'Randomized Search (proxy for Bayesian Opt.)',
                                **{k: str(v) for k, v in best_p.items()},
                                'Best CV Accuracy': round(best_sc, 4)}])

    bo_df.to_csv(os.path.join(script_dir, "Table14_BayesianOpt.csv"), index=False)
    print("Saved Table14_BayesianOpt.csv")
    return bo_df, best_p, best_sc


# ─────────────────────────────────────────────
# 5. Cluster-wise Box-plots
# ─────────────────────────────────────────────
def plot_cluster_boxplots(dfn, script_dir):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(dfn)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km.fit_predict(X_sc)
    dfn_c = dfn.copy()
    dfn_c['Cluster'] = clusters
    dfn_c['Cluster Label'] = dfn_c['Cluster'].map({0: 'Skeptics', 1: 'Pragmatists', 2: 'Enthusiasts'})

    # Pick top 6 columns with highest between-cluster variance
    bv = []
    for col in dfn.columns:
        means = [dfn_c.loc[dfn_c['Cluster'] == k, col].mean() for k in range(3)]
        bv.append((col, np.var(means)))
    top6 = [c for c, _ in sorted(bv, key=lambda x: -x[1])[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    pal = {'Skeptics': '#C44E52', 'Pragmatists': '#4C72B0', 'Enthusiasts': '#55A868'}
    for i, col in enumerate(top6):
        data_plot = [dfn_c.loc[dfn_c['Cluster Label'] == lbl, col].values
                     for lbl in ['Skeptics', 'Pragmatists', 'Enthusiasts']]
        bp = axes[i].boxplot(data_plot, patch_artist=True, notch=False,
                             medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], pal.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_xticklabels(['Skeptics', 'Pragmatists', 'Enthusiasts'], fontsize=8)
        axes[i].set_title(str(col)[:40], fontsize=8, fontweight='bold')
        axes[i].set_ylabel('Likert Score', fontsize=8)

    fig.suptitle('Cluster-Wise Distribution of Key Constructs\n(Red=Skeptics, Blue=Pragmatists, Green=Enthusiasts)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(script_dir, "Figure11_ClusterBoxplots.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data(script_dir)
    dfn = get_likert_matrix(df)

    top_corr           = plot_correlation_heatmap(dfn, script_dir)
    kw_df, chi_df, mw_df = run_significance_tests(df, dfn, script_dir)
    res_df             = run_extended_classifiers(df, script_dir)
    bo_df, best_p, best_sc = run_bayesian_optimisation(df, script_dir)
    plot_cluster_boxplots(dfn, script_dir)

    print("\n\n=== ALL ADVANCED ANALYSES COMPLETE ===")
    print(f"Top correlation: {top_corr.iloc[0].to_dict()}")
    print(f"Best Bayesian Opt CV Accuracy: {best_sc:.4f}")
