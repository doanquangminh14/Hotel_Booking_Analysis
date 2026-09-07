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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.extract import get_project_root
from src.transform import transform_data, load_raw_data, prepare_clustering_data

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

def get_figures_dir() -> Path:
    fig_dir = get_project_root() / "reports" / "eda" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def plot_cancellation_by_hotel_and_segment(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hotel_cancel = df.groupby('hotel')['is_canceled'].mean() * 100
    sns.barplot(x=hotel_cancel.index, y=hotel_cancel.values, ax=axes[0], palette=['#3498db', '#e74c3c'], edgecolor='black')
    axes[0].set_title('Ty le huy phong theo Loai khach san', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Ty le huy (%)')
    axes[0].set_xlabel('Loai khach san')
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    seg_cancel = df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=seg_cancel.values, y=seg_cancel.index, ax=axes[1], palette='viridis', edgecolor='black')
    axes[1].set_title('Ty le huy phong theo Phan khuc thi truong', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Ty le huy (%)')
    axes[1].set_ylabel('Phan khuc thi truong')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_width():.1f}%', (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                         ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    output_path = fig_dir / "01_cancellation_by_hotel_and_segment.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_adr_seasonality_and_trend(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    
    months_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    df_valid = df[df['is_canceled'] == 0]
    adr_monthly = df_valid.groupby(['arrival_date_month', 'hotel'])['adr'].mean().reset_index()
    adr_monthly['arrival_date_month'] = pd.Categorical(adr_monthly['arrival_date_month'], categories=months_order, ordered=True)
    adr_monthly = adr_monthly.sort_values('arrival_date_month')
    
    sns.lineplot(
        data=adr_monthly,
        x='arrival_date_month',
        y='adr',
        hue='hotel',
        marker='o',
        linewidth=2.5,
        palette=['#2980b9', '#e67e22']
    )
    plt.title('Bien dong gia phong trung binh (ADR) theo thang trong nam', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Thang den')
    plt.ylabel('Gia phong trung binh (ADR EUR)')
    plt.xticks(rotation=30)
    plt.legend(title='Loai khach san')
    
    plt.tight_layout()
    output_path = fig_dir / "02_adr_seasonality_and_trend.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_lead_time_vs_cancellation(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    
    lead_summary = df.groupby('lead_time_category')['is_canceled'].agg(['mean', 'count']).reset_index()
    lead_summary['cancel_rate'] = lead_summary['mean'] * 100
    
    ax = sns.barplot(
        data=lead_summary,
        x='lead_time_category',
        y='cancel_rate',
        palette='Blues_r',
        edgecolor='black'
    )
    plt.title('Moi quan he giua Thoi gian dat truoc (Lead Time) va Ty le huy phong', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Nhom thoi gian dat truoc')
    plt.ylabel('Ty le huy phong (%)')
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = fig_dir / "03_lead_time_vs_cancellation.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_deposit_type_cancellation(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    
    deposit_summary = df.groupby('deposit_type')['is_canceled'].agg(['mean', 'count']).reset_index()
    deposit_summary['cancel_rate'] = deposit_summary['mean'] * 100
    
    ax = sns.barplot(
        data=deposit_summary,
        x='deposit_type',
        y='cancel_rate',
        palette=['#27ae60', '#c0392b', '#f39c12'],
        edgecolor='black'
    )
    plt.title('Ty le huy phong theo Chinh sach dat coc (Deposit Type)', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Loai dat coc')
    plt.ylabel('Ty le huy phong (%)')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    plt.tight_layout()
    output_path = fig_dir / "04_deposit_type_cancellation_paradox.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_party_composition_and_spending(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    party_counts = pd.Series({
        'Solo Traveler': df['is_solo_traveler'].sum(),
        'Couple': df['is_couple'].sum(),
        'Family': df['is_family'].sum(),
        'Group': df['is_group'].sum()
    })
    
    axes[0].pie(
        party_counts.values,
        labels=party_counts.index,
        autopct='%1.1f%%',
        colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
        startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    axes[0].set_title('Co cau nhom khach hang (Party Composition)', fontweight='bold', fontsize=12)
    
    df_valid = df[df['is_canceled'] == 0]
    adr_party = {
        'Solo': df_valid[df_valid['is_solo_traveler'] == 1]['adr'].mean(),
        'Couple': df_valid[df_valid['is_couple'] == 1]['adr'].mean(),
        'Family': df_valid[df_valid['is_family'] == 1]['adr'].mean(),
        'Group': df_valid[df_valid['is_group'] == 1]['adr'].mean()
    }
    
    sns.barplot(
        x=list(adr_party.keys()),
        y=list(adr_party.values()),
        ax=axes[1],
        palette=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
        edgecolor='black'
    )
    axes[1].set_title('Gia phong trung binh (ADR) theo Co cau khach', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Gia phong TB (EUR)')
    axes[1].set_xlabel('Nhom khach')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f} EUR', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    output_path = fig_dir / "05_party_composition_and_spending.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_special_requests_vs_cancellation(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    
    req_summary = df.groupby('total_of_special_requests')['is_canceled'].mean().reset_index()
    req_summary['cancel_rate'] = req_summary['is_canceled'] * 100
    
    ax = sns.barplot(
        data=req_summary,
        x='total_of_special_requests',
        y='cancel_rate',
        palette='Greens_r',
        edgecolor='black'
    )
    plt.title('Anh huong cua So luong yeu cau dac biet den Ty le huy phong', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('So luong yeu cau dac biet (Special Requests)')
    plt.ylabel('Ty le huy phong (%)')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.8),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    output_path = fig_dir / "06_special_requests_vs_cancellation.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_countries(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(12, 5))
    
    df_valid = df[df['is_canceled'] == 0]
    top_countries = df_valid['country'].value_counts().head(10).reset_index()
    top_countries.columns = ['country', 'count']
    
    ax = sns.barplot(
        data=top_countries,
        x='country',
        y='count',
        palette='crest',
        edgecolor='black'
    )
    plt.title('Top 10 Quoc gia co luong khach luu tru lon nhat', fontweight='bold', fontsize=13, pad=15)
    plt.xlabel('Ma quoc gia (Country Code)')
    plt.ylabel('So luong don hoan tat')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height() + 200),
                    ha='center', va='bottom', fontsize=9, rotation=20)
                    
    plt.tight_layout()
    output_path = fig_dir / "07_top_countries_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_clustering_personas(df: pd.DataFrame, fig_dir: Path) -> None:
    df_cust, X_scaled, scaler, features = prepare_clustering_data(df)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cust['cluster'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_cust['pca1'] = X_pca[:, 0]
    df_cust['pca2'] = X_pca[:, 1]
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    sample_df = df_cust.sample(min(6000, len(df_cust)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x='pca1',
        y='pca2',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        edgecolor='none',
        ax=ax1
    )
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        s=200, c='black', marker='X', edgecolor='white', linewidth=1.5, label='Centroids'
    )
    ax1.set_title('Khong gian 4 Phan khuc Khach hang (PCA 2D)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend(title='Cum (Cluster)')
    
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    profile_cols = ['lead_time', 'total_stay', 'total_guests', 'adr', 'total_of_special_requests', 'is_repeated_guest']
    cluster_means = df_cust.groupby('cluster')[profile_cols].mean()
    cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)
    
    categories = ['Lead Time', 'Total Stay', 'Guests', 'ADR', 'Special Req', 'Loyal Guest']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.xticks(angles[:-1], categories, color='grey', size=10, fontweight='bold')
    
    cluster_labels = [
        'Cluster 0: Gia dinh cao cap',
        'Cluster 1: Nghi duong dai ngay',
        'Cluster 2: Cap doi tieu chuan',
        'Cluster 3: Khach cong tac & quen'
    ]
    
    for i in range(4):
        values = cluster_norm.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax2.plot(angles, values, linewidth=2, linestyle='solid', label=cluster_labels[i], color=palette[i])
        ax2.fill(angles, values, color=palette[i], alpha=0.12)
        
    ax2.set_title('Radar Chart: Chuc nang va dac tinh 4 Cum', fontweight='bold', fontsize=12, y=1.08)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    output_path = fig_dir / "08_customer_clustering_personas.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_all_eda_reports():
    print("Loading data for EDA analysis...")
    df_raw = load_raw_data()
    df_featured = transform_data(df_raw)
    fig_dir = get_figures_dir()
    
    print("Generating chart 1: Cancellation by Hotel & Segment...")
    plot_cancellation_by_hotel_and_segment(df_featured, fig_dir)
    
    print("Generating chart 2: ADR Seasonality & Monthly Trend...")
    plot_adr_seasonality_and_trend(df_featured, fig_dir)
    
    print("Generating chart 3: Lead Time vs Cancellation...")
    plot_lead_time_vs_cancellation(df_featured, fig_dir)
    
    print("Generating chart 4: Deposit Type Paradox...")
    plot_deposit_type_cancellation(df_featured, fig_dir)
    
    print("Generating chart 5: Party Composition & Spending...")
    plot_party_composition_and_spending(df_featured, fig_dir)
    
    print("Generating chart 6: Special Requests vs Cancellation...")
    plot_special_requests_vs_cancellation(df_featured, fig_dir)
    
    print("Generating chart 7: Top Countries Distribution...")
    plot_top_countries(df_featured, fig_dir)
    
    print("Generating chart 8: Customer Clustering Personas...")
    plot_clustering_personas(df_featured, fig_dir)
    
    print(f"All EDA business insight figures have been saved to: {fig_dir.resolve()}")

if __name__ == "__main__":
    generate_all_eda_reports()
