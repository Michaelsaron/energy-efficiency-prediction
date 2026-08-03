from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_FEATURES = [
    "Relative_Compactness",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
]

ENGINEERED_FEATURES = [
    "Compactness_Height",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates a small number of meaningful engineered features.

    Engineered Features
    -------------------
    Compactness_Height
        Interaction between relative compactness and building height.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def validate(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=BASE_FEATURES)

        missing = [col for col in BASE_FEATURES if col not in X.columns]

        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))

        output = X[BASE_FEATURES].copy()

        for column in BASE_FEATURES:
            output[column] = pd.to_numeric(
                output[column],
                errors="raise",
            )

        return output

    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        output = self.validate(X)

        output["Compactness_Height"] = (
            output["Relative_Compactness"] * output["Overall_Height"]
        )

        return output

    def fit(self, X, y=None) -> "FeatureEngineer":
        transformed = self.create_features(X)

        self.feature_names_in_ = np.asarray(
            BASE_FEATURES,
            dtype=object,
        )

        self.feature_names_out_ = np.asarray(
            transformed.columns,
            dtype=object,
        )

        return self

    def transform(self, X):
        return self.create_features(X)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            return np.asarray(
                BASE_FEATURES + ENGINEERED_FEATURES,
                dtype=object,
            )

        return self.feature_names_out_
