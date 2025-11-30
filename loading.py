import os
import time
from pathlib import Path
from typing import List

import kagglehub
import pandas as pd


DATASET_ID= "chethuhn/network-intrusion-dataset"
DEFAULT_OUTPUT_CSV = Path("data/cyberdataset_concat.csv")


def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mLoading\033[1;34m]\033[0m {message}\033[0m")


def _collect_csv_files(root_path: str) -> List[str]:
    csv_files: List[str] = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def download_and_merge_dataset(output_csv: Path | str, dataset_id: str = DATASET_ID) -> pd.DataFrame:
    """Download the Kaggle dataset, merge CSV files, and persist a single training file.
    Args:
        output_csv: Destination path for the merged CSV.
        dataset_id: Kaggle identifier for the dataset.
    Returns:
        The raw dataframe to be preprocessed.
    """

    start = time.time()
    destination = Path(output_csv)

    _log(f"\033[1;33mDownloading dataset '{dataset_id}' via kagglehub...")
    try:
        download_path = kagglehub.dataset_download(dataset_id)
    except Exception as exc:  # pragma: no cover - depends on Kaggle credentials
        raise RuntimeError("[Dataloader] Unable to download dataset via kagglehub.") from exc

    csv_files = _collect_csv_files(download_path)
    if not csv_files:
        raise FileNotFoundError(f"[Dataloader] No CSV files found in downloaded dataset path: {download_path}")

    _log(f"Found {len(csv_files)} CSV files. Concatenating...")
    # Read and concatenate all CSV files
    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(destination, index=False)
    _log(f"Saved merged dataset to {destination.resolve()}")
    end = time.time()
    _log(f"\033[1;32mDataset load completed in {end - start:.2f} seconds.")
    return merged_df

label_map = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "Infiltration": 9,
    "PortScan": 10,
    "SSH-Patator": 11,
    "Web Attack � Brute Force": 12,
    "Web Attack � Sql Injection": 13,
    "Web Attack � XSS": 14
}

inverse_map = {v: k for k, v in label_map.items()}