# tests/test_preprocessing.py
"""Tests for leakage-safe preprocessing and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    BASE_FEATURES,
    ENGINEERED_FEATURES,
    FeatureEngineer,
)
from src.preprocessing import DataPreprocessor


EXPECTED_ENGINEERED_FEATURES = ["Compactness_Height"]

EXPECTED_FINAL_FEATURE_COUNT = 8


def test_feature_lists_are_current() -> None:
    """The project must use seven raw and one engineered feature."""
    assert len(BASE_FEATURES) == 7
    assert ENGINEERED_FEATURES == EXPECTED_ENGINEERED_FEATURES
    assert len(BASE_FEATURES) + len(ENGINEERED_FEATURES) == 8


def test_feature_engineer_creates_only_one_new_feature() -> None:
    raw = pd.DataFrame(
        [
            {
                "Relative_Compactness": 0.86,
                "Wall_Area": 294.0,
                "Roof_Area": 147.0,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.10,
                "Glazing_Area_Distribution": 1,
            }
        ]
    )

    transformed = FeatureEngineer().create_features(raw)

    assert list(transformed.columns) == BASE_FEATURES + EXPECTED_ENGINEERED_FEATURES
    assert transformed.shape == (1, EXPECTED_FINAL_FEATURE_COUNT)
    assert transformed["Compactness_Height"].iloc[0] == pytest.approx(6.02)


def test_feature_engineer_does_not_modify_input() -> None:
    raw = pd.DataFrame(
        [
            {
                "Relative_Compactness": 0.86,
                "Wall_Area": 294.0,
                "Roof_Area": 147.0,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.10,
                "Glazing_Area_Distribution": 1,
            }
        ]
    )
    original = raw.copy(deep=True)

    FeatureEngineer().create_features(raw)

    pd.testing.assert_frame_equal(raw, original)


def test_preprocessing_split_is_leakage_safe() -> None:
    preprocessor = DataPreprocessor(
        test_size=0.20,
        random_state=42,
        verbose=False,
    )

    X_train, X_test, y_train, y_test = (
        preprocessor.run_preprocessing_pipeline()
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(y_train) + len(y_test)


def test_preprocessing_returns_eight_engineered_columns() -> None:
    preprocessor = DataPreprocessor(verbose=False)

    X_train, X_test, _, _ = preprocessor.run_preprocessing_pipeline()

    expected_columns = BASE_FEATURES + EXPECTED_ENGINEERED_FEATURES

    assert list(X_train.columns) == expected_columns
    assert list(X_test.columns) == expected_columns
    assert X_train.shape[1] == EXPECTED_FINAL_FEATURE_COUNT
    assert X_test.shape[1] == EXPECTED_FINAL_FEATURE_COUNT


def test_preprocessing_is_deterministic() -> None:
    first = DataPreprocessor(random_state=42, verbose=False)
    second = DataPreprocessor(random_state=42, verbose=False)

    X_train_1, X_test_1, y_train_1, y_test_1 = (
        first.run_preprocessing_pipeline()
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = (
        second.run_preprocessing_pipeline()
    )

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)


def test_model_specific_preprocessors_are_fitted_on_training_data() -> None:
    preprocessor = DataPreprocessor(verbose=False)
    X_train, X_test, _, _ = preprocessor.run_preprocessing_pipeline()

    assert preprocessor.is_fitted is True
    assert preprocessor.preprocessor_tree is not None
    assert preprocessor.preprocessor_linear is not None

    tree_train = preprocessor.transform_for_model(
        X_train,
        model_type="tree",
    )
    tree_test = preprocessor.transform_for_model(
        X_test,
        model_type="tree",
    )
    linear_train = preprocessor.transform_for_model(
        X_train,
        model_type="linear",
    )
    linear_test = preprocessor.transform_for_model(
        X_test,
        model_type="linear",
    )

    assert tree_train.shape[0] == len(X_train)
    assert tree_test.shape[0] == len(X_test)
    assert linear_train.shape[0] == len(X_train)
    assert linear_test.shape[0] == len(X_test)

    assert tree_train.shape[1] == tree_test.shape[1]
    assert linear_train.shape[1] == linear_test.shape[1]

    assert np.isfinite(np.asarray(tree_train, dtype=float)).all()
    assert np.isfinite(np.asarray(tree_test, dtype=float)).all()
    assert np.isfinite(np.asarray(linear_train, dtype=float)).all()
    assert np.isfinite(np.asarray(linear_test, dtype=float)).all()


def test_get_processed_data_requires_fitted_pipeline() -> None:
    preprocessor = DataPreprocessor(verbose=False)

    with pytest.raises(ValueError, match="Run run_preprocessing_pipeline"):
        preprocessor.get_processed_data()


def test_missing_raw_feature_raises_clear_error() -> None:
    raw = pd.DataFrame(
        [
            {
                "Relative_Compactness": 0.86,
                "Wall_Area": 294.0,
                "Roof_Area": 147.0,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.10,
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing required"):
        FeatureEngineer().create_features(raw)
