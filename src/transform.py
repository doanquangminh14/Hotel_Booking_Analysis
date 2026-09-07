import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
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

def transform_data(df_raw: pd.DataFrame, save_path: str | Path | None = None) -> pd.DataFrame:
    df_clean = clean_data(df_raw)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path, index=False)
        
    return df_clean

if __name__ == "__main__":
    df_raw = load_raw_data()
    output_path = get_project_root() / "Data" / "hotel_bookings_cleaned.csv"
    df_clean = transform_data(df_raw, save_path=output_path)
    print(f"Data cleaning completed: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    print(f"Saved cleaned data to: {output_path.resolve()}")
