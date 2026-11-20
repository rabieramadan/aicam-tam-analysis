import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, header=1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")
    target_col = None
    for col in df.columns:
        if "attending workshops or training sessions" in str(col).lower():
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
    print(f"Target variable: {target_col}")
    df = df.dropna(subset=[target_col])
    return df, target_col


def prepare_features(df, target_col):
    feature_cols = [c for c in df.columns
                    if c != target_col and "timestamp" not in str(c).lower()]
    X = df[feature_cols].fillna('Unknown').astype(str).copy()
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    le_y = LabelEncoder()
    y = le_y.fit_transform(df[target_col].astype(str))
    classes = le_y.classes_
    return X, y, classes


def evaluate_model(model, X_test, y_test, classes, model_name):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro') if len(classes) > 2 else roc_auc_score(y_test, proba[:, 1])
    except Exception:
        auc = float('nan')
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {'Model': model_name, 'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec, 'ROC_AUC': auc}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, "..", "XLS data.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df, target_col = load_and_prepare_data(data_path)
    if df is None:
        return

    X, y, classes = prepare_features(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Algorithm 1: Random Forest
    # Feature importance via Mean Decrease in Impurity (MDI)
    print("\n--- Algorithm 1: Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, classes, "Random Forest")

    # Algorithm 2: Support Vector Machine (SVM)
    # Primal: min (1/2)||w||^2 + C*sum(xi_i)
    # Dual with RBF Kernel: K(x,z) = exp(-gamma*||x-z||^2)
    print("\n--- Algorithm 2: Support Vector Machine (RBF Kernel) ---")
    svm_pipeline = Pipeline([('scaler', StandardScaler()),
                              ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])
    svm_pipeline.fit(X_train, y_train)
    svm_metrics = evaluate_model(svm_pipeline, X_test, y_test, classes, "SVM (RBF Kernel)")

    # Algorithm 3: Gradient Boosting
    # F_M(x) = sum_{m=1}^{M} gamma_m * h_m(x)
    # Each h_m fits the negative gradient (pseudo-residuals)
    print("\n--- Algorithm 3: Gradient Boosting Classifier ---")
    gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gbc.fit(X_train, y_train)
    gbc_metrics = evaluate_model(gbc, X_test, y_test, classes, "Gradient Boosting")

    # Algorithm 4: Logistic Regression
    # P(y=1|x) = sigma(w^T x + b)
    print("\n--- Algorithm 4: Logistic Regression (Baseline) ---")
    lr_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('lr', LogisticRegression(max_iter=1000, random_state=42))])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, classes, "Logistic Regression")

    # Cross-Validation
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in [("Random Forest", rf), ("SVM", svm_pipeline),
                        ("Gradient Boosting", gbc), ("Logistic Regression", lr_pipeline)]:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_results[name] = scores
        print(f"  {name:25s} Mean Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    # Save comparison table
    all_metrics = [rf_metrics, svm_metrics, gbc_metrics, lr_metrics]
    results_df  = pd.DataFrame(all_metrics).round(4)
    table_path  = os.path.join(script_dir, "Table6_AlgorithmComparison.csv")
    results_df.to_csv(table_path, index=False)
    print(f"\nSaved comparison table to {table_path}")

    # Algorithm comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_cols = ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC_AUC']
    x = np.arange(len(all_metrics))
    width = 0.15
    palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    for i, col in enumerate(metric_cols):
        vals = [m[col] for m in all_metrics]
        ax.bar(x + (i - len(metric_cols)/2)*width + width/2, vals, width, label=col, color=palette[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m['Model'] for m in all_metrics], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Algorithm Comparison: Workshop Attendance Prediction', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure6_AlgorithmComparison.png"), dpi=300)
    plt.close()
    print("Saved Figure6_AlgorithmComparison.png")

    # CV scores chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, (name, scores) in enumerate(cv_results.items()):
        ax2.plot(range(1, 6), scores, marker='o', label=name, color=palette[i], linewidth=2)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('5-Fold CV Accuracy per Fold', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 6))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure7_CrossValidation.png"), dpi=300)
    plt.close()
    print("Saved Figure7_CrossValidation.png")


if __name__ == "__main__":
    main()
