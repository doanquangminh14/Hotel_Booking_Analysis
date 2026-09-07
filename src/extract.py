import os
from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    if filepath is None:
        filepath = get_project_root() / "Data" / "hotel_bookings.csv"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")
    return pd.read_csv(filepath)

def load_featured_data(filepath: str | Path | None = None) -> pd.DataFrame:
    if filepath is None:
        filepath = get_project_root() / "Data" / "hotel_bookings_featured.csv"
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")
    return pd.read_csv(filepath)

def inspect_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "column_names": df.columns.tolist()
    }

if __name__ == "__main__":
    df_raw = load_raw_data()
    summary = inspect_data(df_raw)
    print(f"Loaded raw data: {summary['rows']:,} rows, {summary['columns']} columns")
    print(f"Duplicates: {summary['duplicates']:,}, Missing values: {summary['missing_values']:,}")
