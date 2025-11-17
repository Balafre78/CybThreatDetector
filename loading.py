import os
import time
from pathlib import Path
from typing import List

import kagglehub
import pandas as pd

DATASET_ID = "chethuhn/network-intrusion-dataset"
DEFAULT_OUTPUT_CSV = Path("data/cyberdataset_concat.csv")


def _collect_csv_files(root_path: str) -> List[str]:
    csv_files: List[str] = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def download_and_merge_dataset(
    *,
    dataset_id: str = DATASET_ID,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    force_download: bool = False,
) -> str:
    """Download the Kaggle dataset, merge CSV files, and persist a single training file.

    Args:
        dataset_id: Kaggle identifier for the dataset.
        output_csv: Destination path for the merged CSV.
        force_download: When True, always redownload and rebuild the CSV.

    Returns:
        The string path to the merged CSV on disk.
    """

    start = time.time()
    destination = Path(output_csv)
    if destination.exists() and not force_download:
        print(f"[Dataloader] Using cached dataset at {destination.resolve()}")
        return str(destination)

    print(f"[Dataloader] Downloading dataset '{dataset_id}' via kagglehub...")
    try:
        download_path = kagglehub.dataset_download(dataset_id)
    except Exception as exc:  # pragma: no cover - depends on Kaggle credentials
        raise RuntimeError(
            "[Dataloader] Unable to download dataset via kagglehub."
        ) from exc

    csv_files = _collect_csv_files(download_path)
    if not csv_files:
        raise FileNotFoundError(
            f"[Dataloader] No CSV files found in downloaded dataset path: {download_path}"
        )

    print(f"[Dataloader] Found {len(csv_files)} CSV files. Concatenating...")
    # Read and concatenate all CSV files
    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(destination, index=False)
    print(f"[Dataloader] Saved merged dataset to {destination.resolve()}")
    end = time.time()
    print(f"[Dataloader] Dataset load completed in {end - start:.2f} seconds.")
    return str(destination)