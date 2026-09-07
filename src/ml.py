import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from src.extract import get_project_root
from src.transform import load_raw_data, prepare_clustering_data, transform_data

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

def get_ml_figures_dir() -> Path:
    fig_dir = get_project_root() / "reports" / "ml" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def evaluate_optimal_k(X_scaled: np.ndarray, fig_dir: Path) -> dict:
    k_range = range(2, 9)
    inertias = []
    sil_scores = []
    db_scores = []
    ch_scores = []
    
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        
        sample_labels = labels[sample_idx]
        sil = silhouette_score(X_sample, sample_labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        
        sil_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    axes[0, 0].plot(k_range, inertias, marker='o', color='#2980b9', linewidth=2.5)
    axes[0, 0].set_title('Elbow Method (Inertia / WCSS)', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('So cum (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5)
    
    axes[0, 1].plot(k_range, sil_scores, marker='s', color='#27ae60', linewidth=2.5)
    axes[0, 1].set_title('Silhouette Score (Cao hon la tot hon)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('So cum (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5)
    
    axes[1, 0].plot(k_range, db_scores, marker='^', color='#e67e22', linewidth=2.5)
    axes[1, 0].set_title('Davies-Bouldin Index (Thap hon la tot hon)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('So cum (k)')
    axes[1, 0].set_ylabel('DB Index')
    axes[1, 0].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5)
    
    axes[1, 1].plot(k_range, ch_scores, marker='d', color='#8e44ad', linewidth=2.5)
    axes[1, 1].set_title('Calinski-Harabasz Index (Cao hon la tot hon)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('So cum (k)')
    axes[1, 1].set_ylabel('CH Score')
    axes[1, 1].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    output_path = fig_dir / "01_optimal_k_evaluation.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "sil_scores": sil_scores,
        "db_scores": db_scores,
        "ch_scores": ch_scores
    }

def plot_pca_variance(X_scaled: np.ndarray, fig_dir: Path) -> PCA:
    pca = PCA(n_components=6, random_state=42)
    pca.fit(X_scaled)
    
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    axes[0].bar(range(1, len(exp_var) + 1), exp_var, color='#3498db', edgecolor='black', alpha=0.85)
    axes[0].set_title('Phuong sai giai thich theo tung PC (Scree Plot)', fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Thanh phan chinh (Principal Component)')
    axes[0].set_ylabel('% Phuong sai giai thich')
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                         ha='center', va='bottom', fontsize=9)
                         
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='#e74c3c', linewidth=2.5)
    axes[1].axhline(y=50, color='grey', linestyle='--', alpha=0.7)
    axes[1].set_title('Phuong sai tich luy giai thich (Cumulative Variance)', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('So luong thanh phan chinh')
    axes[1].set_ylabel('% Phuong sai tich luy')
    for x_val, y_val in zip(range(1, len(cum_var) + 1), cum_var):
        axes[1].annotate(f'{y_val:.1f}%', (x_val, y_val + 1.5), ha='center', fontsize=9)
        
    plt.tight_layout()
    output_path = fig_dir / "02_pca_explained_variance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return pca

def plot_pca_2d_clusters(df_cust: pd.DataFrame, kmeans: KMeans, pca: PCA, fig_dir: Path) -> None:
    plt.figure(figsize=(12, 7))
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    sample_df = df_cust.sample(min(8000, len(df_cust)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x='pca1',
        y='pca2',
        hue='cluster',
        palette=palette,
        alpha=0.65,
        s=35,
        edgecolor='none'
    )
    
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        s=250, c='black', marker='X', edgecolor='white', linewidth=2, label='Centroids (Tam cum)'
    )
    
    plt.title('Truc quan hoa 4 Phan khuc Khach hang trong khong gian 2D PCA', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Thanh phan chinh 1 (Principal Component 1)')
    plt.ylabel('Thanh phan chinh 2 (Principal Component 2)')
    plt.legend(title='Phan khuc (Cluster)', loc='upper right')
    
    plt.tight_layout()
    output_path = fig_dir / "03_pca_2d_clusters.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_cluster_feature_distributions(df_cust: pd.DataFrame, fig_dir: Path) -> None:
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    metrics = [
        ('lead_time', 'Thoi gian dat truoc (ngay)'),
        ('total_stay', 'Tong so dem luu tru'),
        ('total_guests', 'So luong khach trung binh'),
        ('adr', 'Gia phong TB / dem (ADR EUR)'),
        ('total_of_special_requests', 'So yeu cau dac biet'),
        ('is_repeated_guest', 'Ty le khach quay lai')
    ]
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        mean_vals = df_cust.groupby('cluster')[col].mean()
        sns.barplot(x=mean_vals.index, y=mean_vals.values, ax=ax, palette=palette, edgecolor='black')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Cum (Cluster)')
        ax.set_ylabel('Gia tri TB')
        for p in ax.patches:
            val_format = f'{p.get_height():.2f}' if p.get_height() < 1 else f'{p.get_height():.1f}'
            ax.annotate(val_format, (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                        
    plt.tight_layout()
    output_path = fig_dir / "04_cluster_feature_distributions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_radar_personas(df_cust: pd.DataFrame, fig_dir: Path) -> None:
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    radar_cols = ['lead_time', 'total_stay', 'total_guests', 'adr', 'total_of_special_requests', 'is_repeated_guest']
    cluster_means = df_cust.groupby('cluster')[radar_cols].mean()
    cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)
    
    categories = ['Lead Time', 'Total Stay', 'Guests', 'ADR', 'Special Req', 'Loyal Guest']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight='bold')
    
    cluster_labels = [
        'Cluster 0: Gia dinh cao cap',
        'Cluster 1: Nghi duong dai ngay',
        'Cluster 2: Cap doi tieu chuan',
        'Cluster 3: Khach cong tac & quen'
    ]
    
    for i in range(4):
        values = cluster_norm.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=cluster_labels[i], color=palette[i])
        ax.fill(angles, values, color=palette[i], alpha=0.15)
        
    plt.title('Radar Chart: So sanh dac tinh 4 Phan khuc Khach hang', size=13, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)
    
    plt.tight_layout()
    output_path = fig_dir / "05_radar_personas.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_ml_pipeline():
    print("Loading data for ML pipeline...")
    df_raw = load_raw_data()
    df_featured = transform_data(df_raw)
    fig_dir = get_ml_figures_dir()
    
    print("Preparing clustering dataset...")
    df_cust, X_scaled, scaler, features = prepare_clustering_data(df_featured)
    
    print("Step 1: Evaluating optimal k (Elbow, Silhouette, DB, CH)...")
    k_metrics = evaluate_optimal_k(X_scaled, fig_dir)
    
    print("Step 2: Performing PCA dimensionality reduction...")
    pca = plot_pca_variance(X_scaled, fig_dir)
    
    print("Step 3: Training KMeans model with k=4...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cust['cluster'] = kmeans.fit_predict(X_scaled)
    
    X_pca = pca.transform(X_scaled)
    df_cust['pca1'] = X_pca[:, 0]
    df_cust['pca2'] = X_pca[:, 1]
    
    print("Step 4: Generating 2D PCA cluster visualization...")
    plot_pca_2d_clusters(df_cust, kmeans, pca, fig_dir)
    
    print("Step 5: Generating cluster feature distributions...")
    plot_cluster_feature_distributions(df_cust, fig_dir)
    
    print("Step 6: Generating Radar personas chart...")
    plot_radar_personas(df_cust, fig_dir)
    
    print(f"ML Pipeline completed successfully. Figures saved to: {fig_dir.resolve()}")

if __name__ == "__main__":
    run_ml_pipeline()
