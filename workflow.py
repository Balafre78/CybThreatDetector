from enum import Enum
from typing import Dict, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from loading import download_and_merge_dataset
from preprocessing import RAW_DATASET_PATH, preprocess_dataset, DEFAULT_TRAIN_DATASET_PATH, DEFAULT_TEST_DATASET_PATH
from testing import evaluate_model
from training import MODELS_DIR, load_models, train_models


class WorkflowStep(Enum):
    LOAD_DATA = 1
    PREPROCESS_DATA = 2
    TRAIN_MODEL = 3
    TEST_MODEL = 4
    @property
    def label(self) -> str:
        return {
            WorkflowStep.LOAD_DATA: "Load data",
            WorkflowStep.PREPROCESS_DATA: "Preprocess data",
            WorkflowStep.TRAIN_MODEL: "Train data",
            WorkflowStep.TEST_MODEL: "Test data",
        }[self]


class WorkflowManager:
    def __init__(self) -> None:
        self.raw_dataset_path = RAW_DATASET_PATH
        self.train_dataset_path = DEFAULT_TRAIN_DATASET_PATH
        self.test_dataset_path = DEFAULT_TEST_DATASET_PATH
        self.models_dir = MODELS_DIR

        self.data_loaded: bool = self.raw_dataset_path.exists()
        self.data_preprocessed: bool = self.train_dataset_path.exists() and self.test_dataset_path.exists()
        self.models_trained: bool = self._has_trained_models()
        self.data_tested: bool = False

        self.models: Optional[XGBClassifier] = None
        self.latest_metrics: Optional[Dict[str, Dict[str, float]]] = {}

    def _has_trained_models(self) -> bool:
        required = ["XGBoost.joblib"]
        return all((self.models_dir / filename).exists() for filename in required)

    def is_step_unlocked(self, step: WorkflowStep) -> bool:
        match step:
            case WorkflowStep.LOAD_DATA:
                return True
            case WorkflowStep.PREPROCESS_DATA:
                return self.data_loaded
            case WorkflowStep.TRAIN_MODEL:
                return self.data_preprocessed
            case WorkflowStep.TEST_MODEL:
                return self.models_trained

    def execute_step(self, step: WorkflowStep):
        if not self.is_step_unlocked(step):
            raise RuntimeError(f"Step '{step.label}' is locked. Complete the previous step first.")
        actions = {
            WorkflowStep.LOAD_DATA: self._run_load_data,
            WorkflowStep.PREPROCESS_DATA: self._run_preprocess_data,
            WorkflowStep.TRAIN_MODEL: self._run_train_models,
            WorkflowStep.TEST_MODEL: self._run_test_models,
        }
        return actions[step]()

    def _run_load_data(self):
        download_and_merge_dataset(output_csv=self.raw_dataset_path)
        self.data_loaded = True

    def _run_preprocess_data(self):
        preprocess_dataset(
            raw_csv=self.raw_dataset_path,
            output_train_csv=self.train_dataset_path,
            output_test_csv=self.test_dataset_path
        )
        self.data_preprocessed = True

    def _run_train_models(self):
        self.models = train_models(train_csv_path=self.train_dataset_path, models_dir=self.models_dir)
        self.models_trained = True

    def _run_test_models(self):
        if self.models is None:
            self.models = load_models(models_dir=self.models_dir)
        model_name = type(self.models).__name__
        metrics = evaluate_model(model=self.models, model_name=model_name, test_csv_path=self.test_dataset_path)
        #self.latest_metrics[model_name] = metrics
        self.data_tested = True


def _format_step_line(index: int, step: WorkflowStep, *, done: bool, unlocked: bool) -> str:
    if done:
        status = "\033[1;32mDone\033[0m"
    elif unlocked:
        status = "\033[1;33mReady\033[0m"
    else:
        status = "\033[1;31mLocked\033[0m"
    return f"\u2502{f' {index}. {step.label}'.ljust(22)}{f'[{status}] \u2502'.rjust(22)}"


def _render_menu(manager: WorkflowManager) -> None:
    print(f'\u250C{32*'\u2500'}\u2510')
    print("\u2502 Cyber Threat Detector Workflow \u2502")
    rows = [
        (WorkflowStep.LOAD_DATA, manager.data_loaded),
        (WorkflowStep.PREPROCESS_DATA, manager.data_preprocessed),
        (WorkflowStep.TRAIN_MODEL, manager.models_trained),
        (WorkflowStep.TEST_MODEL, manager.data_tested),
    ]
    for idx, (step, done) in enumerate(rows, start=1):
        unlocked = manager.is_step_unlocked(step)
        print(_format_step_line(idx, step, done=done, unlocked=unlocked))
    print(f'\u2514{32*'\u2500'}\u2518')
    if manager.latest_metrics:
        print("\nLatest metrics:")
        for model_name, metrics in manager.latest_metrics.items():
            formatted = ", ".join(
                f"{key}={value:.4f}" for key, value in metrics.items()
            )
            print(f" - {model_name}: {formatted}")

def run_cli() -> None:
    manager = WorkflowManager()
    print("Cyber Threat Detector interactive workflow.\nPress Ctrl+C to exit at any time.")
    while True:
        try:
            _render_menu(manager)
            user_choice = input("Enter 1-4 or 'q' to quit ❯ ").strip().lower()
            if user_choice in {"q", "quit", "exit"}:
                print("Exiting workflow. Goodbye!")
                break
            try:
                step = WorkflowStep(int(user_choice))
            except (ValueError, KeyError):
                print("Invalid selection. Please choose a number between 1 and 4 or 'q' to quit.")
                continue
            try:
                manager.execute_step(step)
            except RuntimeError as exc:
                print(f"[Warning] {exc}")
                continue
            except Exception as exc:  # pragma: no cover - surfaces runtime issues to user
                print(f"[Error] {exc}")
                continue
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting workflow.")
            break