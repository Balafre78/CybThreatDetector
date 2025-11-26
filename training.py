from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from loading import label_map

from preprocessing import DEFAULT_TRAIN_DATASET_PATH


MODELS_DIR = Path("models")


def _log(message: str) -> None:
    print(f"\033[1;34m[\033[0;36mTraining\033[1;34m]\033[0m {message}\033[0m")


def _load_train_dataset(
        csv_path: Path | str
) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[Training] Training dataset file not found at {csv_path.resolve()}. Run preprocessing first.")
    _log(f"\033[1;33mLoading training dataset from {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df.iloc[:, :-1], df.iloc[:, -1]


def train_models(
    train_csv_path: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    models_dir: Path | str = MODELS_DIR,
    random_state: int = 39,
) -> XGBClassifier:
    X_train, y_train = _load_train_dataset(train_csv_path)
    y_train = y_train.map(lambda x: label_map[x])
    """
    _log("Training DecisionTreeClassifier...")
    dt_model = DecisionTreeClassifier(max_depth=None, random_state=random_state)
    start = time.time()
    dt_model.fit(X_train, y_train)
    end = time.time()
    _log(f"\033[1;32mDecisionTreeClassifier training completed in {end - start:.2f} seconds.")

    _log("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)
    start = time.time()
    rf_model.fit(X_train, y_train)
    end = time.time()
    _log(f"\033[1;32mRandomForestClassifier training completed in {end - start:.2f} seconds.")

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(dt_model, models_path / "decision_tree.joblib")
    joblib.dump(rf_model, models_path / "random_forest.joblib")
    _log(f"Saved models to {models_path.resolve()}")

    return dt_model, rf_model
    """
    _log("Training XGBoostClassifier...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",   # GPU MODE
    )
    start = time.time()
    # XGBClassifier don't accept string label
    # le = LabelEncoder()
    #model.fit(X_train, le.fit_transform(y_train), verbose=True)
    model.fit(X_train, y_train, verbose=True)
    end = time.time()
    _log(f"\033[1;32mXGBClassifier training completed in {end - start:.2f} seconds.")

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    # joblib.dump(le, models_path / 'label_encoder.joblib')
    joblib.dump(model, models_path / "XGBoost.joblib")
    _log(f"Saved models to {models_path.resolve()}")

    return model


def load_models(models_dir: Path | str = MODELS_DIR) -> XGBClassifier:
    models_path = Path(models_dir)
    _log(f"\033[1;33mLoading training models from {models_path.resolve()}")

    xgboost_path = models_path / "XGBoost.joblib"

    missing = not xgboost_path.exists()
    if missing:
        raise FileNotFoundError("Missing trained models. Expected files: " + ", "+ str(missing))

    model: XGBClassifier = joblib.load(xgboost_path)

    return model

