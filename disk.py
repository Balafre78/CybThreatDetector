from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def _log(message: str, end: str = '\n') -> None:
    print(f"\033[1;34m[\033[0;36mDisk Loading\033[1;34m]\033[0m {message}\033[0m", end=end)


def load_dataset(csv_path: Path | str) -> Optional[pd.DataFrame]:
    """
    Loads a dataset from a CSV file located at the given location
    :param csv_path: Path to load the CSV file from
    :return The loaded Dataframe if successful,none otherwise
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        #raise FileNotFoundError(f"[Disk Loading] Dataset file not found at {csv_path.resolve()}.")
        return None
    _log(f"\033[1;33mLoading dataset from {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def load_model(model_path: Path | str) -> DecisionTreeClassifier | RandomForestClassifier | None:
    """
    Loads a model (DecisionTreeClassifier or RandomForestClassifier) from a given location
    :param model_path: Path to load the model from
    :return The loaded model (DecisionTreeClassifier or RandomForestClassifier) if successful, none otherwise
    """
    models_path = Path(model_path)
    if not model_path.exists():
        #raise FileNotFoundError(f"[Disk Loading] Model file not found at {model_path.resolve()}.")
        return None
    _log(f"\033[1;33mLoading training models from {models_path.resolve()}")
    model = joblib.load(models_path)
    return model
