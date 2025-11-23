from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    print("Loading models...")
    app.decision_tree_model = _load_decision_tree_model(Path("models/decision_tree.joblib"))
    app.random_forest_model = _load_random_forest_model(Path("models/random_forest.joblib"))
    yield
    # Shutdown: nothing to cleanup for now


app = FastAPI(title="CybThreatDetector API", lifespan=lifespan)

# --- CORS configuration ---
# Pour le développement, on autorise toutes les origines. En production, restreindre aux origines nécessaires.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.decision_tree_model: Optional[DecisionTreeClassifier] = None
app.random_forest_model: Optional[RandomForestClassifier] = None


def _load_decision_tree_model(model_path: Path) -> DecisionTreeClassifier:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, DecisionTreeClassifier):
        raise ValueError("Loaded model is not a DecisionTreeClassifier")
    print(f"Loaded model from {model_path.resolve()}")
    return model


def _load_random_forest_model(model_path: Path) -> RandomForestClassifier:
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()}.")
    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Loaded model is not a RandomForestClassifier")
    print(f"Loaded model from {model_path.resolve()}")
    return model


@app.get("/models")
def list_models():
    resp = [{
        "id": "decision_tree",
        "name": "DecisionTreeClassifier",
        "path": "models/decision_tree_model.joblib",
        "loaded": app.decision_tree_model is not None,
    }, {
        "id": "random_forest",
        "name": "RandomForestClassifier",
        "path": "models/random_forest_model.joblib",
        "loaded": app.random_forest_model is not None,
    }]
    return resp


@app.post("/predict/{model_name}")
def predict(model_name: str):
    match model_name:
        case "decision_tree":
            if app.decision_tree_model is None:
                return { "error": "Decision Tree model not loaded." }
            # Dummy prediction logic
            prediction = app.decision_tree_model.predict([[0]*45])
            return { "model": "decision_tree", "prediction": prediction.tolist() }
        case "random_forest":
            if app.random_forest_model is None:
                return { "error": "Random Forest model not loaded." }
            # Dummy prediction logic
            prediction = app.random_forest_model.predict([[0]*45])
            return { "model": "random_forest", "prediction": prediction.tolist() }
    return { "error": "Model not found." }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
