# data_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame and parse dates if possible."""
    df = pd.read_csv(path, parse_dates=True, infer_datetime_format=True, low_memory=False)
    # attempt to convert any column named like 'date'
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def summary(df: pd.DataFrame):
    """Print overall shape, dtypes, memory usage."""
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMemory Usage:")
    print(df.memory_usage(deep=True) / 1024**2, "MB")

def missing_and_duplicates(df: pd.DataFrame):
    """Report missing values and duplicate rows."""
    miss = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values per column:\n", miss[miss > 0])
    dupes = df.duplicated().sum()
    print(f"\nTotal duplicate rows: {dupes}")

def numeric_stats(df: pd.DataFrame):
    """Show descriptive stats for numeric cols."""
    print("\nNumeric summary:")
    print(df.select_dtypes(include=[np.number]).describe().T)

def categorical_summary(df: pd.DataFrame, top_n: int = 10):
    """Show top categories for each object/categorical column."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        print(f"\nColumn: {c}")
        print(df[c].value_counts(dropna=False).head(top_n))

def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5):
    """Return a DataFrame mask of IQR-based outliers for each numeric column."""
    num = df.select_dtypes(include=[np.number])
    outlier_mask = pd.DataFrame(False, index=df.index, columns=num.columns)
    for col in num:
        Q1 = num[col].quantile(0.25)
        Q3 = num[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        mask = (num[col] < lower) | (num[col] > upper)
        outlier_mask[col] = mask
        n_out = mask.sum()
        print(f"{col}: {n_out} outliers by IQR")
    return outlier_mask

def detect_outliers_zscore(df: pd.DataFrame, thresh: float = 3.0):
    """Return mask of Z-score outliers (abs(z) > thresh)."""
    num = df.select_dtypes(include=[np.number])
    zscores = np.abs(stats.zscore(num, nan_policy='omit'))
    mask = (zscores > thresh)
    # Summarize per column
    for i, col in enumerate(num.columns):
        print(f"{col}: {(mask[:, i]).sum()} outliers by Z-score")
    return pd.DataFrame(mask, columns=num.columns, index=df.index)

def plot_distributions(df: pd.DataFrame, cols=None, bins=30):
    """Plot histograms for numeric columns."""
    num = df.select_dtypes(include=[np.number])
    if cols is None:
        cols = num.columns
    for c in cols:
        plt.figure(figsize=(6,4))
        data = num[c].dropna()
        plt.hist(data, bins=bins)
        plt.title(f'Distribution of {c}')
        plt.xlabel(c)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

def correlation_heatmap(df: pd.DataFrame):
    """Plot a simple correlation heatmap."""
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Numeric Feature Correlations')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Initial EDA on a dataset")
    p.add_argument("path", help="Path to your CSV file")
    args = p.parse_args()

    df = load_data(args.path)
    summary(df)
    missing_and_duplicates(df)
    numeric_stats(df)
    categorical_summary(df)
    out_iqr = detect_outliers_iqr(df)
    out_z = detect_outliers_zscore(df)
    # Uncomment the following to see plots
    # plot_distributions(df)
    # correlation_heatmap(df)

# paleidimas
# python data_analysis.py data\master_dataset.csv 