import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

RAW_DATASET_PATH = Path("data/cyberdataset_concat.csv")
DEFAULT_CLEAN_DATASET_PATH = Path("data/cyberdataset_clean.csv")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_raw_dataset(csv_path: Path | str = RAW_DATASET_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[Preprocessing] Raw dataset file not found at {csv_path.resolve()}. Run the data loader first."
        )
    print(f"[Preprocessing] Loading raw dataset from {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def heatmap_correlation(df: pd.DataFrame) -> None:
    """
    Plots a correlation heatmap for all numeric columns in a DataFrame.
    - Automatically filters non-numeric columns
    - Drops rows with NaN values before computing correlations
    """
    numeric_df = df.select_dtypes(include=np.number)
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Adjust figure size dynamically based on number of columns
    n_cols = len(corr_matrix.columns)
    plt.figure(figsize=(min(0.6 * n_cols, 25), min(0.6 * n_cols, 25)))

    # Draw the heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',  # red = positive, blue = negative
        center=0,
        annot=False,  # set to True to show numbers
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": .8}
    )
    # Add title and layout adjustments
    plt.title("Correlation Heatmap", fontsize=30, pad=12)
    plt.tight_layout()
    plt.show()


def clear_df(df: pd.DataFrame) -> pd.DataFrame:
    print("[Preprocessing] Cleaning dataset...")
    # Drop rows with NaN
    print("\tDropping rows with NaN values...")
    cleaned_df = df.dropna()
    #  Remove rows containing +inf or -inf
    print("\tDropping rows with int values...")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number])
    cleaned_df = cleaned_df[~np.isinf(numeric_cols).any(axis=1)]
    cleaned_df = cleaned_df.reset_index(drop=True)
    # Threshold = number of remaining rows
    threshold = len(cleaned_df)
    # Count zero values per column
    zero_counts = (cleaned_df.iloc[:, :-1] == 0).sum()
    #  Columns where all values == 0 and we drop them
    cols_to_drop = zero_counts[zero_counts == threshold].index
    print(f"\tDropping zero full columns ({len(cols_to_drop)})...")
    print(list(cols_to_drop))
    # Drop columns
    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    return cleaned_df


def count_labels(df: pd.DataFrame) -> None:
    label_col = df.columns[-1]  # last column
    counts = df[label_col].value_counts()
    print("\n### Label counts ###")
    print(counts)


def get_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    print(f"[Preprocessing] Searching columns with correlation > {threshold}...")
    # df supposed to be cleared already
    features_only = df.iloc[:, :-1]
    corr_matrix = features_only.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    # Find features that have correlation greater than threshold
    to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]
    return to_drop


def drop_highly_correlated_features(df: pd.DataFrame, high_corr_cols: list[str]) -> pd.DataFrame:
    to_keep = [
        'Subflow Fwd Packets',
        ' Total Backward Packets',
        ' Packet Length Mean',
        ' Fwd IAT Mean',
        ' Fwd Packets/s',
        ' SYN Flag Count',
        ' Fwd Header Length.1',
        ' Idle Mean'
    ]
    # Build the drop list (highly correlated columns not in to_keep)
    to_drop = [col for col in high_corr_cols if col not in to_keep]
    print(f"\tDropping columns ({len(to_drop)})...")
    print(list(to_drop))
    # Drop them
    df_cleaned = df.drop(columns=to_drop, errors='ignore')
    return df_cleaned


def preprocess_dataset(
    *,
    raw_csv: Path | str = RAW_DATASET_PATH,
    output_csv: Path | str = DEFAULT_CLEAN_DATASET_PATH,
) -> str:
    start = time.time()
    df_train = load_raw_dataset(raw_csv)
    df_cleaned = clear_df(df_train)
    #heatmap_correlation(df_cleaned)
    high_corr_cols = get_highly_correlated_features(df_cleaned)
    df_cleaned = drop_highly_correlated_features(df_cleaned, high_corr_cols)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(destination, index=False)
    print(f"[Preprocessing] Saved cleaned dataset to {destination.resolve()}")
    end = time.time()
    print(f"[Preprocessing] Preprocessing completed in {end - start:.2f} seconds.")
    return str(destination)
