import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.extract import get_project_root, load_raw_data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    df_clean = df_clean.drop_duplicates()
    
    df_clean['company'] = df_clean['company'].fillna(0).astype(int)
    df_clean['agent'] = df_clean['agent'].fillna(0).astype(int)
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['children'] = df_clean['children'].fillna(0).astype(int)
    
    df_clean = df_clean[~((df_clean['adults'] == 0) & (df_clean['children'] == 0) & (df_clean['babies'] == 0))]
    
    df_clean['meal'] = df_clean['meal'].replace('Undefined', 'SC')
    df_clean = df_clean[df_clean['market_segment'] != 'Undefined']
    df_clean = df_clean[df_clean['distribution_channel'] != 'Undefined']
    
    df_clean = df_clean[(df_clean['adr'] >= 0) & (df_clean['adr'] < 5000)]
    
    df_clean['reservation_status_date'] = pd.to_datetime(df_clean['reservation_status_date'])
    
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_clean['arrival_date'] = pd.to_datetime(
        df_clean['arrival_date_year'].astype(str) + '-' +
        df_clean['arrival_date_month'].map(month_map).astype(str) + '-' +
        df_clean['arrival_date_day_of_month'].astype(str)
    )
    
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    
    df_fe['total_stay'] = df_fe['stays_in_weekend_nights'] + df_fe['stays_in_week_nights']
    df_fe['weekend_stay_ratio'] = np.where(
        df_fe['total_stay'] > 0,
        df_fe['stays_in_weekend_nights'] / df_fe['total_stay'],
        0.0
    )
    df_fe['is_weekend_arrival'] = df_fe['arrival_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    season_map = {
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
    }
    df_fe['arrival_season'] = df_fe['arrival_date_month'].map(season_map)
    
    df_fe['lead_time_category'] = pd.cut(
        df_fe['lead_time'],
        bins=[-1, 7, 30, 90, 180, 1000],
        labels=['Last-minute (0-7d)', 'Short (8-30d)', 'Medium (31-90d)', 'Long (91-180d)', 'Ultra-long (>180d)']
    )
    
    df_fe['total_guests'] = df_fe['adults'] + df_fe['children'] + df_fe['babies']
    df_fe['is_solo_traveler'] = (df_fe['total_guests'] == 1).astype(int)
    df_fe['is_couple'] = ((df_fe['adults'] == 2) & (df_fe['children'] == 0) & (df_fe['babies'] == 0)).astype(int)
    df_fe['is_family'] = ((df_fe['children'] > 0) | (df_fe['babies'] > 0)).astype(int)
    df_fe['is_group'] = ((df_fe['total_guests'] >= 3) & (df_fe['is_family'] == 0)).astype(int)
    
    df_fe['total_previous_bookings'] = df_fe['previous_cancellations'] + df_fe['previous_bookings_not_canceled']
    df_fe['cancellation_rate_history'] = np.where(
        df_fe['total_previous_bookings'] > 0,
        df_fe['previous_cancellations'] / df_fe['total_previous_bookings'],
        0.0
    )
    
    df_fe['has_special_requests'] = (df_fe['total_of_special_requests'] > 0).astype(int)
    df_fe['has_booking_changes'] = (df_fe['booking_changes'] > 0).astype(int)
    df_fe['has_parking_space'] = (df_fe['required_car_parking_spaces'] > 0).astype(int)
    df_fe['is_room_changed'] = (df_fe['reserved_room_type'] != df_fe['assigned_room_type']).astype(int)
    
    df_fe['adr_per_person'] = np.where(
        df_fe['total_guests'] > 0,
        df_fe['adr'] / df_fe['total_guests'],
        df_fe['adr']
    )
    df_fe['total_cost'] = df_fe['total_stay'] * df_fe['adr']
    df_fe['net_revenue'] = np.where(df_fe['is_canceled'] == 0, df_fe['total_cost'], 0.0)
    df_fe['is_domestic'] = (df_fe['country'] == 'PRT').astype(int)
    
    return df_fe

def prepare_clustering_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, list[str]]:
    if 'total_stay' not in df.columns or 'total_guests' not in df.columns:
        df = engineer_features(df)
        
    df_cust = df[df['is_canceled'] == 0].copy().reset_index(drop=True)
    
    cluster_features = [
        'lead_time',
        'total_stay',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'weekend_stay_ratio',
        'total_guests',
        'is_family',
        'is_solo_traveler',
        'adr',
        'adr_per_person',
        'total_of_special_requests',
        'required_car_parking_spaces',
        'booking_changes',
        'is_repeated_guest'
    ]
    
    X = df_cust[cluster_features].copy()
    
    for col in ['lead_time', 'total_stay', 'adr', 'adr_per_person', 'booking_changes']:
        q99 = X[col].quantile(0.99)
        X[col] = np.where(X[col] > q99, q99, X[col])
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_cust, X_scaled, scaler, cluster_features

def transform_data(df_raw: pd.DataFrame, save_path: str | Path | None = None) -> pd.DataFrame:
    df_clean = clean_data(df_raw)
    df_featured = engineer_features(df_clean)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_featured.to_csv(save_path, index=False)
        
    return df_featured

if __name__ == "__main__":
    df_raw = load_raw_data()
    output_path = get_project_root() / "Data" / "hotel_bookings_featured.csv"
    df_featured = transform_data(df_raw, save_path=output_path)
    print(f"Data transformation completed: {df_featured.shape[0]:,} rows, {df_featured.shape[1]} columns")
    
    df_cust, X_scaled, scaler, features = prepare_clustering_data(df_featured)
    print(f"Clustering dataset prepared: {X_scaled.shape[0]:,} records with {X_scaled.shape[1]} features")
