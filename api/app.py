from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd
from io import StringIO


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    print("Loading models...")
    try:
        app.decision_tree_model = _load_decision_tree_model(Path("models/decision_tree.joblib"))
    except Exception as e:
        print(e)
        app.decision_tree_model = None
    try:
        app.random_forest_model = _load_random_forest_model(Path("models/random_forest.joblib"))
    except Exception as e:
        print(e)
        app.random_forest_model = None
    try:
        app.logistic_regression_model = _load_logistic_regression_model(Path("models/logistic_regression.joblib"))
    except Exception as e:
        print(e)
        app.logistic_regression_model = None
    try:
        app.xgboost_model = _load_xgboost_model(Path("models/xgboost.joblib"))
    except Exception as e:
        print(e)
        app.xgboost_model = None
    print("Models loaded!")
    yield
    # Shutdown: nothing to cleanup for now


app = FastAPI(title="CybThreatDetector API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.decision_tree_model: Optional[DecisionTreeClassifier] = None
app.random_forest_model: Optional[RandomForestClassifier] = None
app.logistic_regression_model: Optional[LogisticRegression] = None
app.xgboost_model: Optional[XGBClassifier] = None


def _load_decision_tree_model(model_path: Path) -> DecisionTreeClassifier:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"DecisionTreeClassifier model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, DecisionTreeClassifier):
        raise ValueError("Loaded model is not a DecisionTreeClassifier")
    print(f"Loaded model from {model_path.resolve()}")
    return model


def _load_random_forest_model(model_path: Path) -> RandomForestClassifier:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"RandomForestClassifier model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Loaded model is not a RandomForestClassifier")
    print(f"Loaded model from {model_path.resolve()}")
    return model

def _load_logistic_regression_model(model_path: Path) -> LogisticRegression:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"LogisticRegression model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise ValueError("Loaded model is not a LogisticRegression")
    print(f"Loaded model from {model_path.resolve()}")
    return model

def _load_xgboost_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"XGBClassifier model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, XGBClassifier):
        raise ValueError("Loaded model is not a XGBClassifier")
    print(f"Loaded model from {model_path.resolve()}")
    return model


@app.get("/models")
def list_models():
    resp = [{
        "id": "decision_tree",
        "name": "DecisionTreeClassifier",
        "loaded": app.decision_tree_model is not None,
    }, {
        "id": "random_forest",
        "name": "RandomForestClassifier",
        "loaded": app.random_forest_model is not None,
    }, {
        "id": "logistic_regression",
        "name": "LogisticRegression",
        "loaded": app.logistic_regression_model is not None,
    }, {
        "id": "xgboost",
        "name": "XGBClassifier",
        "loaded": app.xgboost_model is not None,
    }]
    return resp


def _get_model(model_name: str):
    match model_name:
        case "decision_tree":
            return app.decision_tree_model
        case "random_forest":
            return app.random_forest_model
        case "logistic_regression":
            return app.logistic_regression_model
        case "xgboost":
            return app.xgboost_model
        case _:
            return None


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    model = _get_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found or not loaded.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    X_test, y_test = df.iloc[:, :-1], df.iloc[:, -1]

    try:
        preds = model.predict(X_test)
        if model_name == "xgboost":
            preds = pd.Series(preds).map(lambda x: inverse_map[x])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return preds.tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
