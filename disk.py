from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def _log(message: str, end: str = '\n') -> None:
    print(f"\033[1;34m[\033[0;36mDisk Loading\033[1;34m]\033[0m {message}\033[0m", end=end)


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """
    TODO
    Args:
        csv_path: TODO
    Returns:
        TODO
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[Disk Loading] Dataset file not found at {csv_path.resolve()}.")
    _log(f"\033[1;33mLoading dataset from {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def load_model(model_path: Path | str) -> DecisionTreeClassifier | RandomForestClassifier:
    """
    TODO
    Args:
        model_path: TODO
    Returns:
        TODO
    """
    models_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"[Disk Loading] Model file not found at {model_path.resolve()}.")
    _log(f"\033[1;33mLoading training models from {models_path.resolve()}")
    model = joblib.load(models_path)
    return model
