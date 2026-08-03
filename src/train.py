from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.feature_engineering import BASE_FEATURES
from src.model_adapters import CatBoostSklearnAdapter
from src.preprocessing import DataPreprocessor
from src.visualization import create_comparison_chart
from src.utils import (
    METRICS_DIR,
    environment_metadata,
    estimator_name,
    save_json,
    save_model,
)

RANDOM_STATE = 42
CV_FOLDS = 5


MODEL_COMPLEXITY = {
    "Linear Regression": "Low",
    "Ridge Regression": "Low",
    "Lasso Regression": "Low",
    "Decision Tree": "Medium",
    "Random Forest": "High",
    "Gradient Boosting": "High",
    "SVR": "Medium",
    "XGBoost": "High",
    "Extra Trees": "High",
    "AdaBoost": "Medium",
    "CatBoost": "High",
    "LightGBM": "High",
}


def _normalised_benefit(series: pd.Series) -> pd.Series:
    """Scale a higher-is-better metric to the 0–1 range."""
    span = float(series.max() - series.min())
    if span == 0:
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / span


def _normalised_cost(series: pd.Series) -> pd.Series:
    """Scale a lower-is-better metric to the 0–1 range."""
    span = float(series.max() - series.min())
    if span == 0:
        return pd.Series(1.0, index=series.index)
    return 1.0 - ((series - series.min()) / span)


def _generalisation_status(gap: float) -> str:
    gap = abs(float(gap))
    if gap <= 0.01:
        return "Strong"
    if gap <= 0.03:
        return "Good"
    if gap <= 0.07:
        return "Moderate"
    return "Possible overfitting"


def _stability_status(cv_std: float) -> str:
    cv_std = abs(float(cv_std))
    if cv_std <= 0.005:
        return "Very stable"
    if cv_std <= 0.015:
        return "Stable"
    if cv_std <= 0.035:
        return "Moderate"
    return "Variable"



def scaled(estimator):
    return Pipeline(
        [
            (
                "preprocessing",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
            ("model", estimator),
        ]
    )


def unscaled(estimator):
    return Pipeline(
        [
            (
                "preprocessing",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
            ),
            ("model", estimator),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    models = {
        "Linear Regression": scaled(LinearRegression()),
        "Ridge Regression": scaled(Ridge(alpha=1.0)),
        "Lasso Regression": scaled(
            Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": unscaled(
            DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)
        ),
        "Random Forest": unscaled(
            RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            )
        ),
        "Gradient Boosting": unscaled(
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
            )
        ),
        "SVR": scaled(SVR(kernel="rbf", C=100, epsilon=0.05)),
        "Extra Trees": unscaled(
            ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        ),
        "AdaBoost": unscaled(
            AdaBoostRegressor(
                n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
            )
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = unscaled(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = unscaled(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401

        models["CatBoost"] = unscaled(
            CatBoostSklearnAdapter(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )
        )
    except ImportError:
        pass
    return models


class ModelTrainer:
    def __init__(self):
        self.models = build_models()
        self.fitted_models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None

    @staticmethod
    def metrics(y_train, p_train, y_test, p_test):
        train_mse = mean_squared_error(y_train, p_train)
        test_mse = mean_squared_error(y_test, p_test)
        train_r2 = r2_score(y_train, p_train)
        test_r2 = r2_score(y_test, p_test)
        return {
            "Train MAE": mean_absolute_error(y_train, p_train),
            "Train MSE": train_mse,
            "Train RMSE": np.sqrt(train_mse),
            "Train R2": train_r2,
            "MAE": mean_absolute_error(y_test, p_test),
            "MSE": test_mse,
            "RMSE": np.sqrt(test_mse),
            "R2": test_r2,
            "Train-Test R2 Gap": train_r2 - test_r2,
        }

    def train_all_models(self, X_train, y_train, X_test, y_test):
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        failures: dict[str, str] = {}

        for name, template in self.models.items():
            try:
                model = clone(template)
                start = time.perf_counter()
                model.fit(X_train, y_train)
                elapsed = time.perf_counter() - start

                p_train = model.predict(X_train)
                p_test = model.predict(X_test)

                row = {
                    "Algorithm": name,
                    **self.metrics(y_train, p_train, y_test, p_test),
                }

                scores = cross_validate(
                    clone(template),
                    X_train,
                    y_train,
                    cv=cv,
                    scoring={
                        "r2": "r2",
                        "mae": "neg_mean_absolute_error",
                        "rmse": "neg_root_mean_squared_error",
                    },
                    n_jobs=-1,
                    error_score="raise",
                )

                row.update(
                    {
                        "CV R2 Mean": scores["test_r2"].mean(),
                        "CV R2 Std": scores["test_r2"].std(),
                        "CV MAE Mean": -scores["test_mae"].mean(),
                        "CV RMSE Mean": -scores["test_rmse"].mean(),
                        "Training Time (s)": elapsed,
                    }
                )

                self.results.append(row)
                self.fitted_models[name] = model
                print(
                    f"{name}: R²={row['R2']:.4f}, "
                    f"RMSE={row['RMSE']:.4f}"
                )

            except Exception as exc:
                failures[name] = str(exc)
                print(f"{name}: skipped because training failed: {exc}")

        if not self.results:
            raise RuntimeError(
                "No regression model trained successfully. "
                "Check the installed packages and dataset."
            )

        if failures:
            failure_path = METRICS_DIR / "model_failures.csv"
            pd.DataFrame(
                [
                    {"Algorithm": name, "Failure": message}
                    for name, message in failures.items()
                ]
            ).to_csv(failure_path, index=False)
            print(f"Model failure log saved: {failure_path}")

        results = pd.DataFrame(self.results)

        # Balanced deployment score: accuracy is important, but the winner is
        # not selected from R² alone. Error, CV performance, CV variability,
        # and the train-test gap are also considered.
        results["Deployment Score"] = 100 * (
            0.40 * _normalised_benefit(results["R2"])
            + 0.15 * _normalised_cost(results["RMSE"])
            + 0.10 * _normalised_cost(results["MAE"])
            + 0.20 * _normalised_benefit(results["CV R2 Mean"])
            + 0.075 * _normalised_cost(results["CV R2 Std"])
            + 0.075 * _normalised_cost(results["Train-Test R2 Gap"].abs())
        )

        results["Generalisation"] = results["Train-Test R2 Gap"].map(
            _generalisation_status
        )
        results["Stability"] = results["CV R2 Std"].map(_stability_status)
        results["Complexity"] = results["Algorithm"].map(MODEL_COMPLEXITY).fillna(
            "Unknown"
        )

        results = results.sort_values(
            ["Deployment Score", "R2", "RMSE"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        results.insert(0, "Deployment Rank", np.arange(1, len(results) + 1))
        self.best_model_name = str(results.iloc[0]["Algorithm"])
        self.best_model = self.fitted_models[self.best_model_name]
        self.results = results.to_dict("records")
        return results

    def save_outputs(self, X_train, X_test, target_col):
        results = pd.DataFrame(self.results)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
        results.to_csv(METRICS_DIR / "deployment_ranking.csv", index=False)

        # Save labelled comparison figures for the report and presentation.
        create_comparison_chart(results, save=True, show=False)

        winner = results.iloc[0]
        save_model(self.best_model, "best_model.pkl")
        metadata = {
            "best_model": self.best_model_name,
            "estimator_class": estimator_name(self.best_model),
            "selection_rule": (
                "Balanced deployment score: 40% Test R2, 15% RMSE, "
                "10% MAE, 20% CV R2 mean, 7.5% CV variability, and "
                "7.5% train-test gap."
            ),
            "best_r2": float(winner["R2"]),
            "best_rmse": float(winner["RMSE"]),
            "best_mae": float(winner["MAE"]),
            "train_r2": float(winner["Train R2"]),
            "train_test_r2_gap": float(winner["Train-Test R2 Gap"]),
            "cv_r2_mean": float(winner["CV R2 Mean"]),
            "cv_r2_std": float(winner["CV R2 Std"]),
            "training_time_seconds": float(winner["Training Time (s)"]),
            "deployment_score": float(winner["Deployment Score"]),
            "generalisation_status": str(winner["Generalisation"]),
            "stability_status": str(winner["Stability"]),
            "model_complexity": str(winner["Complexity"]),
            "unseen_test_metrics_source": "outputs/metrics/unseen_test_predictions.csv",
            "selection_justification": (
                f"{self.best_model_name} ranked first using the balanced deployment "
                f"score ({float(winner['Deployment Score']):.2f}/100). It combines "
                f"held-out accuracy, prediction error, cross-validation performance, "
                f"cross-validation stability and the train-test gap."
            ),
            "environment": environment_metadata(),
        }
        save_json(metadata, "model_metadata.json")

        # Save current feature definitions dynamically so metadata cannot
        # become stale when a raw feature is added or removed.
        save_json(
            {
                "raw_feature_names": BASE_FEATURES.copy(),
                "feature_names": X_train.columns.tolist(),
                "n_raw_features": len(BASE_FEATURES),
                "n_features": X_train.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "target": target_col,
                "feature_engineering": True,
                "preprocessing_saved_inside_model": True,
            },
            "feature_info.json",
        )
        print(f"Authoritative deployment model: {self.best_model_name}")
        return results


def run_training_pipeline(target_col="Heating_Load"):
    pre = DataPreprocessor(target_col=target_col, apply_feature_engineering=True)
    X_train, X_test, y_train, y_test = pre.run_preprocessing_pipeline()
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    results = trainer.save_outputs(X_train, X_test, target_col)
    return trainer, results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_training_pipeline()
