import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import re

def clean_likert_value(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r'^\d', str(val).strip())
    if match:
        return float(match.group())
    return np.nan

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, "..", "data", "survey_anonymised.csv")

    print("Loading data...")
    try:
        df = pd.read_csv(data_path, header=0)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Detect Likert columns
    likert_cols = []
    for col in df.columns:
        non_nulls = df[col].dropna().astype(str)
        if len(non_nulls) == 0:
            continue
        if non_nulls.str.match(r'^\d').mean() > 0.2:
            likert_cols.append(col)

    print(f"Found {len(likert_cols)} Likert-scale columns.")

    df_num = pd.DataFrame(index=df.index)
    for col in likert_cols:
        df_num[col] = df[col].apply(clean_likert_value)
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())
    df_num = df_num.dropna(axis=1, how='all')

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    # -------------------------------------------------------
    # Optimal K via Elbow Method + Silhouette + Davies-Bouldin
    # WCSS: J(C) = sum_{k=1}^{K} sum_{x in C_k} ||x - mu_k||^2
    # -------------------------------------------------------
    print("\nRunning Elbow Method (k=2..8)...")
    K_range = range(2, 9)
    wcss_vals, sil_vals, db_vals, ch_vals = [], [], [], []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(X_scaled, labels))
        db_vals.append(davies_bouldin_score(X_scaled, labels))
        ch_vals.append(calinski_harabasz_score(X_scaled, labels))
        print(f"  k={k} | WCSS={km.inertia_:.1f} | Silhouette={sil_vals[-1]:.4f} | "
              f"Davies-Bouldin={db_vals[-1]:.4f} | Calinski-Harabasz={ch_vals[-1]:.1f}")

    # Save cluster validation table
    cv_df = pd.DataFrame({
        'K': list(K_range),
        'WCSS': wcss_vals,
        'Silhouette': sil_vals,
        'Davies_Bouldin': db_vals,
        'Calinski_Harabasz': ch_vals
    })
    cv_df.to_csv(os.path.join(script_dir, "Table7_ClusterValidation.csv"), index=False)
    print("Saved Table7_ClusterValidation.csv")

    # Elbow / Silhouette plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(K_range), wcss_vals, marker='o', color='steelblue', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0].set_ylabel('WCSS (Inertia)', fontsize=11)
    axes[0].set_title('Elbow Method: Optimal K Selection', fontsize=13, fontweight='bold')
    axes[0].axvline(3, color='red', linestyle='--', linewidth=1.5, label='K=3 (selected)')
    axes[0].legend()

    axes[1].plot(list(K_range), sil_vals, marker='s', color='darkorange', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score vs K', fontsize=13, fontweight='bold')
    axes[1].axvline(3, color='red', linestyle='--', linewidth=1.5, label='K=3 (selected)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure3b_ElbowSilhouette.png"), dpi=300)
    plt.close()
    print("Saved Figure3b_ElbowSilhouette.png")

    # -------------------------------------------------------
    # Final K-Means with K=3
    # -------------------------------------------------------
    n_clusters = 3
    print(f"\nRunning KMeans with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster']     = clusters
    df_num['Cluster'] = clusters

    sil_final = silhouette_score(X_scaled, clusters)
    db_final  = davies_bouldin_score(X_scaled, clusters)
    print(f"  Final Silhouette Score : {sil_final:.4f}")
    print(f"  Final Davies-Bouldin   : {db_final:.4f}")
    print(f"  Final WCSS             : {kmeans.inertia_:.2f}")

    # Cluster profiles
    for i in range(n_clusters):
        size = (clusters == i).sum()
        print(f"\n--- Cluster {i} (N={size}) ---")
        cmeans = df_num[df_num['Cluster'] == i].drop('Cluster', axis=1).mean()
        overall = df_num.drop('Cluster', axis=1).mean()
        diffs = cmeans - overall
        print("  Top positive diffs:", diffs.nlargest(3).to_dict())
        print("  Top negative diffs:", diffs.nsmallest(3).to_dict())

    # -------------------------------------------------------
    # PCA: Covariance matrix eigendecomposition
    # C = (1/(n-1)) X^T X  →  C v_k = lambda_k v_k
    # Explained variance ratio: lambda_k / sum(lambda)
    # -------------------------------------------------------
    pca = PCA()
    X_pca_all = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)

    # How many PCs to explain >=80% variance?
    n_80 = np.searchsorted(cumulative_evr, 0.80) + 1
    print(f"\nPCA: {n_80} components explain {cumulative_evr[n_80-1]:.2%} of variance.")
    print(f"PC1 explains {evr[0]:.2%}, PC2 explains {evr[1]:.2%}")

    # Save PCA variance summary
    pca_df = pd.DataFrame({
        'PC': [f"PC{i+1}" for i in range(len(evr))],
        'ExplainedVariance': np.round(evr, 6),
        'CumulativeVariance': np.round(cumulative_evr, 6)
    })
    pca_df.to_csv(os.path.join(script_dir, "Table8_PCA_Variance.csv"), index=False)
    print("Saved Table8_PCA_Variance.csv")

    # Scree plot
    n_show = min(15, len(evr))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n_show + 1), evr[:n_show], color='steelblue', alpha=0.7, label='Explained Variance')
    ax.plot(range(1, n_show + 1), cumulative_evr[:n_show], marker='o', color='darkorange', linewidth=2, label='Cumulative Variance')
    ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, label='80% threshold')
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Proportion of Variance', fontsize=11)
    ax.set_title('PCA Scree Plot: Explained Variance by Component', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure3c_PCA_Scree.png"), dpi=300)
    plt.close()
    print("Saved Figure3c_PCA_Scree.png")

    # 2D PCA cluster scatter
    X_pca_2d = X_pca_all[:, :2]
    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    labels_map = {0: 'Skeptics', 1: 'Pragmatists', 2: 'Enthusiastic Adopters'}
    for ci in range(n_clusters):
        mask = clusters == ci
        plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    c=colors[ci], label=f"Cluster {ci}: {labels_map.get(ci, '')}",
                    alpha=0.75, s=70, edgecolors='white', linewidths=0.3)
    plt.title('Student Profiles (K-Means, K=3) – PCA Projection', fontsize=15, fontweight='bold')
    plt.xlabel(f'PC1 ({evr[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({evr[1]:.1%} variance)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "cluster_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved cluster_plot.png")

    # HAC Dendrogram
    linkage_matrix = linkage(X_scaled[:100], method='ward')
    plt.figure(figsize=(14, 7))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=15, leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Agglomerative Clustering – Ward Linkage Dendrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index (or cluster size)', fontsize=11)
    plt.ylabel('Ward Distance (Euclidean)', fontsize=11)
    plt.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='Cut point → K=3')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "Figure3_Dendrogram.png"), dpi=300)
    plt.close()
    print("Saved Figure3_Dendrogram.png")

if __name__ == '__main__':
    main()
