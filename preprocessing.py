import time
from typing import Tuple

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_DATASET_PATH = Path("data/cyberdataset_raw.csv")
DEFAULT_TRAIN_DATASET_PATH = Path("data/cyberdataset_train.csv")
DEFAULT_TEST_DATASET_PATH = Path("data/cyberdataset_test.csv")


def _log(message: str, end: str = '\n') -> None:
    print(f"\033[1;34m[\033[0;36mPreprocessing\033[1;34m]\033[0m {message}\033[0m", end=end)


def _clear_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO
    Args:
        df: TODO
    Returns:
        TODO
    """
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
    """
    TODO
    Args:
        df: TODO
    """
    label_col = df.columns[-1]  # last column
    counts = df[label_col].value_counts()
    _log("Label distribution:")
    print(counts)


def _get_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """
    TODO
    Args:
        df: TODO
        threshold: TODO
    Returns:
        TODO
    """
    _log(f"Searching columns with correlation > {threshold}...")
    # df supposed to be cleared already
    features_only = df.iloc[:, :-1]
    corr_matrix = features_only.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features that have correlation greater than threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return to_drop


def _drop_highly_correlated_features(df: pd.DataFrame, high_corr_cols: list[str]) -> pd.DataFrame:
    """
    TODO
    Args:
        df: TODO
        high_corr_cols: TODO
    Returns:
        TODO
    """
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

def _split_df(df: pd.DataFrame, label_col, test_size: float = 0.2, random_state: int = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO
    Args:
        df: TODO
        label_col: TODO
        test_size: TODO
        random_state: TODO
    Returns:
        TODO
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def _oversample_df(df: pd.DataFrame, label_col, min_samples: int = 2000, random_state: int = None) -> pd.DataFrame:
    """
    TODO
    Args:
        df: TODO
        label_col: TODO
        min_samples: TODO
        random_state: TODO
    Returns:
        TODO
    """
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
    df_raw: pd.DataFrame,
    test_size: float = 0.2,
    output_train_csv: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    output_test_csv: Path | str = DEFAULT_TEST_DATASET_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO
    Args:
        df_raw: TODO
        test_size: TODO
        output_train_csv: TODO
        output_test_csv: TODO
    Returns:
        TODO
    """
    start = time.time()
    #df_raw = _load_raw_dataset(raw_csv)
    df_cleaned = _clear_df(df_raw)
    #heatmap_correlation(df_cleaned)
    high_corr_cols = _get_highly_correlated_features(df_cleaned)
    df_cleaned = _drop_highly_correlated_features(df_cleaned, high_corr_cols)
    df_train, df_test = _split_df(df_cleaned, df_cleaned.columns[-1],test_size=test_size)
    df_train = _oversample_df(df_train, df_cleaned.columns[-1])

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
    return df_train, df_test
