import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def clean_likert(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r'^\d', str(val).strip())
    if match:
        return float(match.group())
    return np.nan

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "survey_anonymised.csv")

    print("Loading data...")
    try:
        df_raw = pd.read_csv(data_path, header=None)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Identify the target column for Ethical Concerns (Academic integrity)
    target_col_idx = -1
    for i in range(df_raw.shape[1]):
        col_text = " ".join(df_raw.iloc[0:1, i].dropna().astype(str))
        if "Academic integrity" in col_text and "plagiarism" in col_text:
            target_col_idx = i
            break

    if target_col_idx == -1:
        print("Target column not found.")
        return

    df = pd.read_csv(data_path, header=0)
    target_col = df.columns[target_col_idx]

    print(f"Target Column: {target_col}")

    # Process target
    df['Target_Concern'] = df[target_col].apply(clean_likert)
    df = df.dropna(subset=['Target_Concern'])

    # Process features
    feature_cols = []
    for col in df.columns:
        if col != target_col and col != 'Target_Concern' and "Timestamp" not in str(col):
            feature_cols.append(col)

    X = pd.DataFrame(index=df.index)

    for col in feature_cols:
        non_nulls = df[col].dropna().astype(str)
        if len(non_nulls) == 0:
            continue

        digit_starts = non_nulls.str.match(r'^\d')
        if digit_starts.mean() > 0.2:
            X[col] = df[col].apply(clean_likert)
            X[col] = X[col].fillna(X[col].median())
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))

    y = df['Target_Concern']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------------------------------
    # Step 1: Baseline Random Forest (default parameters)
    # -------------------------------------------------------
    print("\n--- Step 1: Baseline Random Forest ---")
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_base = rf_baseline.predict(X_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base  = r2_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)

    print(f"  Baseline MSE : {mse_base:.4f}")
    print(f"  Baseline MAE : {mae_base:.4f}")
    print(f"  Baseline R²  : {r2_base:.4f}")

    # -------------------------------------------------------
    # Step 2: Hyperparameter Optimization via GridSearchCV
    # Mathematical basis:
    #   Objective: minimize E[L(y, f(x))] = MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    #   over the parameter space Θ = {n_estimators, max_depth, min_samples_split}
    #   GridSearchCV exhaustively evaluates all combinations using 5-fold CV.
    # -------------------------------------------------------
    print("\n--- Step 2: Hyperparameter Optimization (GridSearchCV) ---")
    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
    }

    rf_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    rf_search.fit(X_train, y_train)

    best_params = rf_search.best_params_
    rf_optimized = rf_search.best_estimator_

    print(f"\n  Best Parameters Found: {best_params}")

    y_pred_opt = rf_optimized.predict(X_test)
    mse_opt = mean_squared_error(y_test, y_pred_opt)
    r2_opt  = r2_score(y_test, y_pred_opt)
    mae_opt = mean_absolute_error(y_test, y_pred_opt)

    print(f"\n  Optimized MSE : {mse_opt:.4f}")
    print(f"  Optimized MAE : {mae_opt:.4f}")
    print(f"  Optimized R²  : {r2_opt:.4f}")

    # Cross-Validation Score
    cv_scores = cross_val_score(rf_optimized, X, y, cv=5, scoring='r2')
    print(f"\n  5-Fold Cross-Validation R² Scores: {np.round(cv_scores, 4)}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save optimization comparison summary
    summary_path = os.path.join(script_dir, "Table5_Optimization.csv")
    summary_df = pd.DataFrame({
        'Model': ['Baseline RF (default)', f'Optimized RF (GridSearchCV)'],
        'n_estimators': [100, best_params.get('n_estimators', 'N/A')],
        'max_depth': ['None', best_params.get('max_depth', 'N/A')],
        'MSE': [round(mse_base, 4), round(mse_opt, 4)],
        'MAE': [round(mae_base, 4), round(mae_opt, 4)],
        'R2': [round(r2_base, 4), round(r2_opt, 4)],
        'Mean_CV_R2': ['N/A', round(cv_scores.mean(), 4)],
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved optimization summary to {summary_path}")

    # -------------------------------------------------------
    # Step 3: Feature Importance (from optimized model)
    # -------------------------------------------------------
    importances = rf_optimized.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = 10
    top_features   = [X.columns[i] for i in indices[:top_n]]
    top_importances = [importances[i] for i in indices[:top_n]]

    plt.figure(figsize=(12, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]
    formatted_features = [str(f)[:55] + ("..." if len(str(f)) > 55 else "") for f in top_features[::-1]]

    bars = plt.barh(range(top_n), top_importances[::-1], color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(range(top_n), formatted_features, fontsize=9)
    plt.title('Top 10 Drivers of Ethical Concern (Optimized Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Decrease in Impurity (Feature Importance)', fontsize=11)
    plt.axvline(x=np.mean(top_importances), color='navy', linestyle='--', linewidth=1.2, label=f'Mean Importance = {np.mean(top_importances):.4f}')
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(script_dir, "Figure4_FeatureImportance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved feature importance plot to {plot_path}")

    # -------------------------------------------------------
    # Step 4: Optimization Comparison Bar Chart
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_labels = ['MSE (lower=better)', 'MAE (lower=better)', 'R² (higher=better)']
    baseline_vals  = [mse_base, mae_base, r2_base]
    optimized_vals = [mse_opt,  mae_opt,  r2_opt]

    x = np.arange(len(metrics_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, baseline_vals,  width, label='Baseline RF',   color='#5B8DB8', edgecolor='black')
    bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized RF',  color='#E07B39', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_title('Baseline vs Optimized Random Forest Performance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.legend()
    ax.bar_label(bars1, fmt='%.4f', padding=2, fontsize=8)
    ax.bar_label(bars2, fmt='%.4f', padding=2, fontsize=8)
    plt.tight_layout()
    comp_path = os.path.join(script_dir, "Figure5_Optimization_Comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved optimization comparison chart to {comp_path}")

    print("\nTop 3 Driving Factors (Optimized Model):")
    for i in range(3):
        print(f"  {i+1}. {top_features[i]}  (importance={top_importances[i]:.4f})")

if __name__ == '__main__':
    main()
