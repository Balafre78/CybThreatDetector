import time
from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from pandas import Index
from sklearn.model_selection import train_test_split

RAW_DATASET_PATH = Path("data/cyberdataset_raw.csv")
DEFAULT_TRAIN_DATASET_PATH = Path("data/cyberdataset_train.csv")
DEFAULT_TEST_DATASET_PATH = Path("data/cyberdataset_test.csv")

def _log(message: str, end: str = '\n') -> None:
    print(f"\033[1;34m[\033[0;36mPreprocessing\033[1;34m]\033[0m {message}\033[0m", end=end)

def _load_raw_dataset(csv_path: Path | str = RAW_DATASET_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[Preprocessing] Raw dataset file not found at {csv_path.resolve()}. Run the data loader first.")
    _log(f"\033[1;33mLoading raw dataset from {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def _heatmap_correlation(df: pd.DataFrame) -> None:
    """
    Plots a correlation heatmap for all numeric columns in a DataFrame.
    - Automatically filters non-numeric columns
    - Drops rows with NaN values before computing correlations
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

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


def _clear_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN
    _log("Dropping rows with NaN values...")
    cleaned_df = df.dropna()
    #  Remove rows containing +inf or -inf
    _log("Dropping rows with int values...")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number])
    cleaned_df = cleaned_df[~np.isinf(numeric_cols).any(axis=1)]
    cleaned_df = cleaned_df.reset_index(drop=True)
    # Threshold = number of remaining rows
    threshold = len(cleaned_df)
    # Count zero values per column
    zero_counts = (cleaned_df.iloc[:, :-1] == 0).sum()
    #  Columns where all values == 0 and we drop them
    cols_to_drop = zero_counts[zero_counts == threshold].index
    _log(f"Dropping zero full columns ({len(cols_to_drop)})...")
    print(list(cols_to_drop))
    # Drop columns
    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    return cleaned_df


def _count_labels(df: pd.DataFrame) -> None:
    label_col = df.columns[-1]  # last column
    counts = df[label_col].value_counts()
    _log("Label distribution:")
    print(counts)


def _get_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    _log(f"Searching columns with correlation > {threshold}...")
    # df supposed to be cleared already
    features_only = df.iloc[:, :-1]
    corr_matrix = features_only.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features that have correlation greater than threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return to_drop


def _drop_highly_correlated_features(df: pd.DataFrame, high_corr_cols: list[str]) -> pd.DataFrame:
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
    _log(f"Dropping columns ({len(to_drop)})...")
    print(list(to_drop))
    # Drop them
    df_cleaned = df.drop(columns=to_drop, errors='ignore')
    return df_cleaned

def _split_df(
        df: pd.DataFrame,
        label_col: Index,
        test_size: float = 0.2,
        random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def _oversample_df(
        df: pd.DataFrame,
        label_col: Index,
        min_samples: int = 2000,
        random_state: int = None
) -> pd.DataFrame:
    np.random.seed(random_state)
    df_balanced = df.copy()
    new_rows = []
    value_counts = df_balanced[label_col].value_counts()
    print("\n=== Oversampling Summary ===")
    for label, count in value_counts.items():
        if count < min_samples:
            needed = min_samples - count
            print(f"Class '{label}': {count} → {min_samples}   (adding {needed})")
            class_rows = df_balanced[df_balanced[label_col] == label]
            duplicated_rows = class_rows.sample(
                n=needed,
                replace=True,
                random_state=random_state
            )
            new_rows.append(duplicated_rows)
    if new_rows:
        df_balanced = pd.concat([df_balanced] + new_rows, ignore_index=True)
    # Shuffle the dataframe
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_balanced


def preprocess_dataset(
    raw_csv: Path | str = RAW_DATASET_PATH,
    test_size: float = 0.2,
    output_train_csv: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    output_test_csv: Path | str = DEFAULT_TEST_DATASET_PATH
):
    start = time.time()
    df_raw = _load_raw_dataset(raw_csv)
    df_cleaned = _clear_df(df_raw)
    #heatmap_correlation(df_cleaned)
    high_corr_cols = _get_highly_correlated_features(df_cleaned)
    df_cleaned = _drop_highly_correlated_features(df_cleaned, high_corr_cols)
    df_train, df_test = _split_df(df_cleaned, label_col=df_cleaned.columns[-1], test_size=test_size)
    df_train = _oversample_df(df_train, label_col=df_cleaned.columns[-1])

    dest_train = Path(output_train_csv)
    dest_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(dest_train, index=False)
    _log(f"Saved training dataset to {dest_train.resolve()}")

    dest_test = Path(output_test_csv)
    dest_test.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(dest_test, index=False)
    _log(f"Saved testing dataset to {dest_test.resolve()}")

    _count_labels(df_train)
    end = time.time()
    _log(f"\033[1;32mPreprocessing completed in {end - start:.2f} seconds.")
