from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for path in (DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR):
    path.mkdir(parents=True, exist_ok=True)


def load_data(
    file_name: str = "energy_efficiency_data.csv", *, verbose: bool = False
) -> pd.DataFrame:
    path = Path(file_name)
    candidates = (
        [path]
        if path.is_absolute()
        else [DATA_DIR / path, PROJECT_ROOT / path, Path.cwd() / "data" / path]
    )
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            df.columns = [str(c).strip() for c in df.columns]
            if df.empty:
                raise ValueError(f"Dataset is empty: {candidate}")
            if verbose:
                print(
                    f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {candidate}"
                )
            return df
    raise FileNotFoundError(
        f"Dataset not found. Checked: {[str(p) for p in candidates]}"
    )


def _model_path(name: str | Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else MODELS_DIR / path


def save_model(
    model: Any, model_name: str = "best_model.pkl", *, verbose: bool = True
) -> Path:
    path = _model_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    if verbose:
        print(f"Saved model: {path}")
    return path


def load_model(model_name: str = "best_model.pkl", *, verbose: bool = False) -> Any:
    path = _model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run: python -m src.train")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            "The saved model is incompatible with the active Python environment. "
            "Retrain it in this environment with: python -m src.train"
        ) from exc
    if verbose:
        print(f"Loaded model: {path}")
    return model


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_json(data: Any, file_name: str, *, verbose: bool = True) -> Path:
    path = _model_path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    if verbose:
        print(f"Saved JSON: {path}")
    return path


def load_json(file_name: str, *, required: bool = False) -> Any | None:
    path = _model_path(file_name)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required metadata file was not found: {path}. "
                "Run: python -m src.train"
            )
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Metadata file is not valid JSON: {path}. "
            "Regenerate it with: python -m src.train"
        ) from exc


def unwrap_estimator(model: Any) -> Any:
    """Return the final estimator from a fitted sklearn Pipeline when possible."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def estimator_name(model: Any) -> str:
    return type(unwrap_estimator(model)).__name__


def environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
