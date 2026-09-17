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

CLUSTER_PALETTE = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

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
    axes[0, 0].set_xlabel('Số cụm (k)')
    axes[0, 0].set_ylabel('Inertia (Tổng bình phương khoảng cách nội cụm)')
    axes[0, 0].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5, label='k=4 (Điểm chọn)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(k_range, sil_scores, marker='s', color='#27ae60', linewidth=2.5)
    axes[0, 1].set_title('Silhouette Score (Độ tách biệt & gắn kết)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Số cụm (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5, label='k=4 (Điểm chọn)')
    axes[0, 1].legend()
    
    axes[1, 0].plot(k_range, db_scores, marker='^', color='#e67e22', linewidth=2.5)
    axes[1, 0].set_title('Davies-Bouldin Index (Giá trị nhỏ hơn là tốt hơn)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Số cụm (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5, label='k=4 (Điểm chọn)')
    axes[1, 0].legend()
    
    axes[1, 1].plot(k_range, ch_scores, marker='d', color='#8e44ad', linewidth=2.5)
    axes[1, 1].set_title('Calinski-Harabasz Index (Giá trị lớn hơn là tốt hơn)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Số cụm (k)')
    axes[1, 1].set_ylabel('Calinski-Harabasz Score')
    axes[1, 1].axvline(x=4, color='#c0392b', linestyle='--', linewidth=1.5, label='k=4 (Điểm chọn)')
    axes[1, 1].legend()
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    
    axes[0].bar(range(1, len(exp_var) + 1), exp_var, color='#3498db', edgecolor='black', alpha=0.85)
    axes[0].set_title('Phương sai giải thích theo từng PC (Scree Plot)', fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Thành phần chính (Principal Component)')
    axes[0].set_ylabel('% Phương sai giải thích')
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                         ha='center', va='bottom', fontsize=9)
                         
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='#e74c3c', linewidth=2.5)
    axes[1].axhline(y=50, color='grey', linestyle='--', alpha=0.7, label='Ngưỡng 50% phương sai')
    axes[1].set_title('Phương sai tích lũy giải thích (Cumulative Variance)', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Số lượng thành phần chính')
    axes[1].set_ylabel('% Phương sai tích lũy')
    axes[1].legend()
    for x_val, y_val in zip(range(1, len(cum_var) + 1), cum_var):
        axes[1].annotate(f'{y_val:.1f}%', (x_val, y_val + 1.5), ha='center', fontsize=9)
        
    plt.tight_layout()
    output_path = fig_dir / "02_pca_explained_variance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return pca

def plot_pca_loadings(pca: PCA, features: list[str], fig_dir: Path) -> pd.DataFrame:
    loadings_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=features
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar=True, linewidths=0.5)
    plt.title('Ma trận trọng số đặc trưng PCA (Principal Component Loadings Matrix)', fontweight='bold', fontsize=12, pad=12)
    plt.xlabel('Thành phần chính')
    plt.ylabel('Đặc trưng mô hình (Features)')
    plt.tight_layout()
    
    output_path = fig_dir / "06_pca_feature_loadings.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return loadings_df

def plot_pca_2d_clusters(df_cust: pd.DataFrame, kmeans: KMeans, pca: PCA, fig_dir: Path) -> None:
    plt.figure(figsize=(12, 7.5))
    
    sample_df = df_cust.sample(min(8000, len(df_cust)), random_state=42)
    cluster_names = {
        0: 'Cụm 0: Gia đình cao cấp',
        1: 'Cụm 1: Nghỉ dưỡng dài ngày',
        2: 'Cụm 2: Cặp đôi tiêu chuẩn',
        3: 'Cụm 3: Khách công tác & quen'
    }
    sample_df['cluster_name'] = sample_df['cluster'].map(cluster_names)
    
    sns.scatterplot(
        data=sample_df,
        x='pca1',
        y='pca2',
        hue='cluster_name',
        palette=CLUSTER_PALETTE,
        alpha=0.65,
        s=35,
        edgecolor='none'
    )
    
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        s=250, c='black', marker='X', edgecolor='white', linewidth=2, label='Centroids (Tâm cụm)'
    )
    
    plt.title('Trực quan hóa 4 Phân khúc Khách hàng trong không gian 2D PCA', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Thành phần chính 1 (PC1 - Thời lượng & Đặt trước)')
    plt.ylabel('Thành phần chính 2 (PC2 - Đoàn khách & Mức chi tiêu ADR)')
    plt.legend(title='Phân khúc (Cluster)', loc='upper right')
    
    plt.tight_layout()
    output_path = fig_dir / "03_pca_2d_clusters.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_cluster_feature_distributions(df_cust: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    metrics = [
        ('lead_time', 'Thời gian đặt trước (ngày)'),
        ('total_stay', 'Tổng số đêm lưu trú'),
        ('total_guests', 'Số lượng khách trung bình'),
        ('adr', 'Giá phòng TB / đêm (ADR EUR)'),
        ('total_of_special_requests', 'Số yêu cầu đặc biệt TB'),
        ('is_repeated_guest', 'Tỷ lệ khách quen quay lại')
    ]
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        mean_vals = df_cust.groupby('cluster')[col].mean()
        sns.barplot(x=mean_vals.index, y=mean_vals.values, hue=mean_vals.index, ax=ax, palette=CLUSTER_PALETTE, edgecolor='black', legend=False)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Cụm (Cluster)')
        ax.set_ylabel('Giá trị TB')
        for p in ax.patches:
            val_format = f'{p.get_height():.2f}' if p.get_height() < 1 else f'{p.get_height():.1f}'
            ax.annotate(val_format, (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                        
    plt.tight_layout()
    output_path = fig_dir / "04_cluster_feature_distributions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_radar_personas(df_cust: pd.DataFrame, fig_dir: Path) -> None:
    radar_cols = ['lead_time', 'total_stay', 'total_guests', 'adr', 'total_of_special_requests', 'is_repeated_guest']
    cluster_means = df_cust.groupby('cluster')[radar_cols].mean()
    cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)
    
    categories = ['Lead Time', 'Total Stay', 'Guests', 'ADR', 'Special Req', 'Loyal Guest']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='#333333', size=11, fontweight='bold')
    
    cluster_labels = [
        'Cluster 0: Gia đình cao cấp',
        'Cluster 1: Cặp đôi tiêu chuẩn',
        'Cluster 2: Nghỉ dưỡng dài ngày',
        'Cluster 3: Khách công tác & quen'
    ]
    
    for i in range(4):
        values = cluster_norm.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=cluster_labels[i], color=CLUSTER_PALETTE[i])
        ax.fill(angles, values, color=CLUSTER_PALETTE[i], alpha=0.15)
        
    plt.title('Radar Chart: So sánh đặc tính 4 Phân khúc Khách hàng', size=13, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)
    
    plt.tight_layout()
    output_path = fig_dir / "05_radar_personas.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_cluster_business_metrics(df_cust: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Doanh thu tổng hợp theo cụm
    df_cust['total_revenue'] = df_cust['total_stay'] * df_cust['adr']
    rev_by_cluster = df_cust.groupby('cluster')['total_revenue'].sum()
    rev_pct = rev_by_cluster / rev_by_cluster.sum() * 100
    
    axes[0].pie(
        rev_pct.values,
        labels=[f'Cụm {i}' for i in rev_pct.index],
        autopct='%1.1f%%',
        colors=CLUSTER_PALETTE,
        startangle=140,
        wedgeprops=dict(edgecolor='black', linewidth=1)
    )
    axes[0].set_title('Tỷ trọng đóng góp Doanh thu theo Cụm', fontweight='bold', fontsize=11)
    
    # 2. Giá trị đơn đặt phòng trung bình (Average Booking Value)
    avg_booking_val = df_cust.groupby('cluster')['total_revenue'].mean()
    sns.barplot(x=avg_booking_val.index, y=avg_booking_val.values, hue=avg_booking_val.index, ax=axes[1], palette=CLUSTER_PALETTE, edgecolor='black', legend=False)
    axes[1].set_title('Giá trị đơn đặt trung bình (EUR / Booking)', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Cụm (Cluster)')
    axes[1].set_ylabel('Giá trị trung bình (EUR)')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f} EUR', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                         
    # 3. Tỷ lệ đơn đặt qua OTA theo cụm
    ota_ratio = df_cust.groupby('cluster')['market_segment'].apply(lambda s: (s == 'Online TA').mean() * 100)
    sns.barplot(x=ota_ratio.index, y=ota_ratio.values, hue=ota_ratio.index, ax=axes[2], palette=CLUSTER_PALETTE, edgecolor='black', legend=False)
    axes[2].set_title('Tỷ lệ đặt phòng qua Online TA (%)', fontweight='bold', fontsize=11)
    axes[2].set_xlabel('Cụm (Cluster)')
    axes[2].set_ylabel('Tỷ lệ (%)')
    for p in axes[2].patches:
        axes[2].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')
                         
    plt.tight_layout()
    output_path = fig_dir / "07_cluster_business_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_cluster_summary_table(df_cust: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    summary_list = []
    for col in features:
        mean_s = df_cust.groupby('cluster')[col].mean()
        median_s = df_cust.groupby('cluster')[col].median()
        std_s = df_cust.groupby('cluster')[col].std()
        
        for c in range(4):
            summary_list.append({
                'Feature': col,
                'Cluster': c,
                'Mean': round(mean_s[c], 3),
                'Median': round(median_s[c], 3),
                'Std': round(std_s[c], 3)
            })
            
    summary_df = pd.DataFrame(summary_list)
    output_path = get_project_root() / "reports" / "ml" / "cluster_profiles.csv"
    summary_df.to_csv(output_path, index=False)
    return summary_df

def run_ml_pipeline():
    print("Loading data for ML pipeline...")
    df_raw = load_raw_data()
    df_featured = transform_data(df_raw)
    fig_dir = get_ml_figures_dir()
    
    print("Preparing clustering dataset...")
    df_cust, X_scaled, scaler, features = prepare_clustering_data(df_featured)
    
    print("Step 1: Evaluating optimal k (Elbow, Silhouette, DB, CH)...")
    k_metrics = evaluate_optimal_k(X_scaled, fig_dir)
    
    print("Step 2: Performing PCA dimensionality reduction & Scree Plot...")
    pca = plot_pca_variance(X_scaled, fig_dir)
    
    print("Step 3: Generating PCA Feature Loadings matrix...")
    loadings_df = plot_pca_loadings(pca, features, fig_dir)
    
    print("Step 4: Training KMeans model with k=4...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cust['cluster'] = kmeans.fit_predict(X_scaled)
    
    X_pca = pca.transform(X_scaled)
    df_cust['pca1'] = X_pca[:, 0]
    df_cust['pca2'] = X_pca[:, 1]
    
    print("Step 5: Generating 2D PCA cluster visualization...")
    plot_pca_2d_clusters(df_cust, kmeans, pca, fig_dir)
    
    print("Step 6: Generating cluster feature distributions...")
    plot_cluster_feature_distributions(df_cust, fig_dir)
    
    print("Step 7: Generating Radar personas chart...")
    plot_radar_personas(df_cust, fig_dir)
    
    print("Step 8: Generating Cluster Business & Revenue Metrics chart...")
    plot_cluster_business_metrics(df_cust, fig_dir)
    
    print("Step 9: Exporting detailed cluster statistical summary table...")
    summary_df = generate_cluster_summary_table(df_cust, features)
    
    print(f"ML Pipeline completed successfully. 7 figures and summary table saved to: {fig_dir.parent.resolve()}")

if __name__ == "__main__":
    run_ml_pipeline()
