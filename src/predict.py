from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.feature_engineering import BASE_FEATURES, FeatureEngineer
from src.utils import estimator_name, load_json, load_model


class EnergyPredictor:
    def __init__(self):
        self.model = load_model("best_model.pkl")
        self.feature_info = load_json("feature_info.json", required=True)
        self.metadata = load_json("model_metadata.json", required=True)
        self.engineer = FeatureEngineer(verbose=False)
        actual = estimator_name(self.model)
        expected = self.metadata.get("estimator_class")

        saved_raw_features = self.feature_info.get("raw_feature_names", [])
        if list(saved_raw_features) != list(BASE_FEATURES):
            raise RuntimeError(
                "Feature metadata does not match the current project inputs. "
                "Run python -m src.train again."
            )

        expected_feature_count = self.feature_info.get("n_features")
        saved_feature_names = self.feature_info.get("feature_names", [])
        if expected_feature_count != len(saved_feature_names):
            raise RuntimeError(
                "feature_info.json contains inconsistent feature metadata. "
                "Run python -m src.train again."
            )
        if expected and expected != actual:
            raise RuntimeError(
                f"Model metadata mismatch: metadata says {expected}, pickle contains {actual}. "
                "Run python -m src.train again."
            )

    def prepare(
        self, data: Mapping[str, Any] | pd.DataFrame | Sequence[float]
    ) -> pd.DataFrame:
        if isinstance(data, Mapping):
            missing = [key for key in BASE_FEATURES if key not in data]
            if missing:
                raise ValueError(
                    "Missing required prediction inputs: " + ", ".join(missing)
                )
            raw = pd.DataFrame([{key: data[key] for key in BASE_FEATURES}])
        elif isinstance(data, pd.DataFrame):
            raw = data.copy()
        else:
            values = list(data)
            if len(values) != len(BASE_FEATURES):
                raise ValueError(f"Expected {len(BASE_FEATURES)} values.")
            raw = pd.DataFrame([values], columns=BASE_FEATURES)
        engineered = self.engineer.create_features(raw)
        expected = self.feature_info["feature_names"]
        missing = [c for c in expected if c not in engineered.columns]
        if missing:
            raise ValueError("Missing engineered features: " + ", ".join(missing))
        return engineered[expected]

    def predict(self, data) -> np.ndarray:
        prepared = self.prepare(data)
        return np.asarray(self.model.predict(prepared), dtype=float)

    def predict_one(self, data) -> float:
        return float(self.predict(data)[0])


_predictor = None


def get_predictor() -> EnergyPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnergyPredictor()
    return _predictor


def predict_heating_load(data) -> float:
    return get_predictor().predict_one(data)


if __name__ == "__main__":
    sample = {
        "Relative_Compactness": 0.86,
        "Wall_Area": 294.0,
        "Roof_Area": 147.0,
        "Overall_Height": 7.0,
        "Orientation": 2,
        "Glazing_Area": 0.10,
        "Glazing_Area_Distribution": 1,
    }
    predictor = EnergyPredictor()
    print(predictor.metadata["best_model"], predictor.predict_one(sample))
