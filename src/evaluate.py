# src/evaluate.py
"""
Evaluation module for the trained energy-efficiency model.

The module:
- evaluates training and held-out test performance;
- saves predictions from unseen test data;
- creates evaluation and residual plots;
- performs SHAP analysis when supported;
- performs cross-validation;
- saves evaluation metrics without regenerating written reports.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)

from src.feature_engineering import BASE_FEATURES, FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.utils import (
    FIGURES_DIR,
    MODELS_DIR,
    load_data,
    load_json,
    load_model,
)
from src.visualization import create_model_evaluation_plots, plot_feature_importance

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

UNSEEN_PREDICTIONS_PATH = METRICS_DIR / "unseen_test_predictions.csv"

EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"


class ModelEvaluator:
    """
    Evaluate the saved production model.

    Parameters
    ----------
    model_path:
        Name of the saved model file inside the models folder.
    """

    def __init__(
        self,
        model_path: str = "best_model.pkl",
    ) -> None:
        self.model = load_model(model_path)
        self.model_name = model_path.replace(".pkl", "")

        self.engineer = FeatureEngineer()

        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

        self.feature_names: list[str] = []

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load the raw input features and heating-load target.
        """
        feature_info = load_json("feature_info.json")

        if isinstance(feature_info, dict):
            self.feature_names = feature_info.get(
                "feature_names",
                [],
            )

        dataframe = load_data()

        if "Heating_Load" not in dataframe.columns:
            raise ValueError(
                "The dataset does not contain the 'Heating_Load' target column."
            )

        missing = [column for column in BASE_FEATURES if column not in dataframe.columns]
        if missing:
            raise ValueError(
                "The dataset is missing required model inputs: "
                + ", ".join(missing)
            )

        # Use exactly the same seven raw features used during training.
        X = dataframe[BASE_FEATURES].copy()

        y = pd.to_numeric(
            dataframe["Heating_Load"],
            errors="raise",
        )

        return X, y

    def prepare_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the same feature engineering used during training.

        The FeatureEngineer always starts from the current raw
        feature list, so calling it here does not duplicate engineered
        columns.
        """
        return self.engineer.create_features(X)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the saved model on training and unseen test data.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        y_pred_train = np.asarray(
            self.model.predict(X_train),
            dtype=float,
        )

        y_pred_test = np.asarray(
            self.model.predict(X_test),
            dtype=float,
        )

        y_train_values = np.asarray(
            y_train,
            dtype=float,
        )

        y_test_values = np.asarray(
            y_test,
            dtype=float,
        )

        train_mse = mean_squared_error(
            y_train_values,
            y_pred_train,
        )

        test_mse = mean_squared_error(
            y_test_values,
            y_pred_test,
        )

        metrics = {
            "Train MAE": float(
                mean_absolute_error(
                    y_train_values,
                    y_pred_train,
                )
            ),
            "Train MSE": float(train_mse),
            "Train RMSE": float(np.sqrt(train_mse)),
            "Train R2": float(
                r2_score(
                    y_train_values,
                    y_pred_train,
                )
            ),
            "Test MAE": float(
                mean_absolute_error(
                    y_test_values,
                    y_pred_test,
                )
            ),
            "Test MSE": float(test_mse),
            "Test RMSE": float(np.sqrt(test_mse)),
            "Test R2": float(
                r2_score(
                    y_test_values,
                    y_pred_test,
                )
            ),
        }

        metrics["R2 Train-Test Gap"] = float(metrics["Train R2"] - metrics["Test R2"])

        print("\nPerformance metrics:")

        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")

        print("\nCreating evaluation plots...")

        create_model_evaluation_plots(
            y_test_values,
            y_pred_test,
            self.model_name,
            save=True,
            show=False,
        )

        print("Evaluation plots saved successfully.")

        self.residual_analysis(
            y_test_values,
            y_pred_test,
        )

        self.save_unseen_test_predictions(
            y_test=y_test_values,
            y_pred_test=y_pred_test,
            test_indices=X_test.index,
        )

        self.save_metrics(metrics)

        return (
            metrics,
            y_pred_train,
            y_pred_test,
        )

    def save_unseen_test_predictions(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        test_indices: pd.Index,
    ) -> pd.DataFrame:
        """
        Save predictions made only on the held-out test set.

        These records were not used to fit the model.
        """
        results = pd.DataFrame(
            {
                "Dataset_Index": list(test_indices),
                "Actual": y_test,
                "Predicted": y_pred_test,
            }
        )

        results["Residual"] = results["Actual"] - results["Predicted"]

        results["Absolute_Error"] = results["Residual"].abs()

        results["Squared_Error"] = results["Residual"] ** 2

        results.to_csv(
            UNSEEN_PREDICTIONS_PATH,
            index=False,
        )

        print("\nSaved unseen test predictions:")
        print(f"   {UNSEEN_PREDICTIONS_PATH}")
        print(f"   Test records: {len(results)}")

        return results

    @staticmethod
    def save_metrics(
        metrics: dict[str, float],
    ) -> None:
        """
        Save numerical evaluation metrics as JSON.
        """
        # FIXED: Added parentheses here to correctly open the Path object
        with EVALUATION_METRICS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

        print("\nSaved evaluation metrics:")
        print(f"   {EVALUATION_METRICS_PATH}")

    def residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Analyse residual errors on held-out test data.
        """
        residuals = y_true - y_pred

        residual_series = pd.Series(
            residuals,
            dtype=float,
        )

        print("\nResidual analysis on unseen data:")
        print(f"   Mean residual: {residual_series.mean():.4f}")
        print(f"   Residual standard deviation: {residual_series.std():.4f}")
        print(f"   Skewness: {residual_series.skew():.4f}")
        print(f"   Kurtosis: {residual_series.kurtosis():.4f}")

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
        )

        axes[0].scatter(
            y_pred,
            residuals,
            alpha=0.6,
        )
        axes[0].axhline(
            y=0,
            linestyle="--",
        )
        axes[0].set_xlabel("Predicted heating load")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Unseen Test Residuals vs Predicted")

        sns.histplot(
            residuals,
            kde=True,
            ax=axes[1],
        )
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Unseen Test Residual Distribution")

        figure.tight_layout()

        output_path = FIGURES_DIR / "unseen_test_residual_analysis.png"

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

        print(f"   Residual figure saved: {output_path}")

        return residuals

    @staticmethod
    def _unwrap_estimator(model: Any) -> Any:
        """
        Return the final estimator when the saved model is a pipeline.
        """
        if hasattr(model, "steps") and model.steps:
            return model.steps[-1][1]

        if hasattr(model, "named_steps") and model.named_steps:
            return list(model.named_steps.values())[-1]

        return model

    def shap_analysis(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Any | None:
        """
        Generate SHAP plots when supported by the saved model.
        """
        print("\n" + "=" * 60)
        print("SHAP EXPLAINABILITY ANALYSIS")
        print("=" * 60)

        estimator = self._unwrap_estimator(self.model)

        try:
            background = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42,
            )

            try:
                explainer = shap.Explainer(
                    estimator,
                    background,
                )
                explanation = explainer(X_test)
                shap_values = explanation
            except Exception:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(12, 8))

            shap.summary_plot(
                shap_values,
                X_test,
                show=False,
            )

            plt.title("SHAP Feature Importance Summary")
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / "shap_summary.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(10, 8))

            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="bar",
                show=False,
            )

            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / "shap_bar.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            print("SHAP analysis completed successfully.")

            return shap_values

        except Exception as exc:
            print(f"SHAP analysis could not be generated for this model: {exc}")

            return None

    def feature_importance_analysis(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Save feature importance for the selected model.

        Native tree importance is preferred. Permutation importance on the
        held-out test set is used as a model-agnostic fallback.
        """
        estimator = self._unwrap_estimator(self.model)
        feature_names = list(X_test.columns)

        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
            method = "native"
        elif hasattr(estimator, "coef_"):
            importance = np.abs(np.ravel(estimator.coef_)).astype(float)
            method = "absolute coefficient"
        else:
            result = permutation_importance(
                self.model,
                X_test,
                y_test,
                scoring="r2",
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            importance = np.asarray(result.importances_mean, dtype=float)
            method = "permutation"

        if len(importance) != len(feature_names):
            print("Feature importance skipped: feature count mismatch.")
            return pd.DataFrame()

        importance_df = plot_feature_importance(
            importance,
            feature_names,
            self.model_name,
            top_n=min(15, len(feature_names)),
            save=True,
            show=False,
        )
        importance_df["Method"] = method
        importance_df.to_csv(
            METRICS_DIR / "feature_importance.csv",
            index=False,
        )
        print(f"Feature importance saved using {method} importance.")
        return importance_df

    def learning_curve_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Save a learning curve for data-size and overfitting analysis."""
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        sizes, train_scores, validation_scores = learning_curve(
            self.model,
            X,
            y,
            cv=cv,
            scoring="r2",
            train_sizes=np.linspace(0.20, 1.00, 5),
            n_jobs=-1,
        )

        curve = pd.DataFrame(
            {
                "Training Samples": sizes,
                "Train R2 Mean": train_scores.mean(axis=1),
                "Train R2 Std": train_scores.std(axis=1),
                "Validation R2 Mean": validation_scores.mean(axis=1),
                "Validation R2 Std": validation_scores.std(axis=1),
            }
        )
        curve.to_csv(METRICS_DIR / "learning_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(
            sizes,
            curve["Train R2 Mean"],
            marker="o",
            label="Training R²",
        )
        ax.plot(
            sizes,
            curve["Validation R2 Mean"],
            marker="o",
            label="Validation R²",
        )
        ax.fill_between(
            sizes,
            curve["Validation R2 Mean"] - curve["Validation R2 Std"],
            curve["Validation R2 Mean"] + curve["Validation R2 Std"],
            alpha=0.15,
        )
        ax.set(
            xlabel="Training samples",
            ylabel="R²",
            title="Learning Curve: Training vs Validation Performance",
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / "learning_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("Learning curve saved.")
        return curve

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> np.ndarray:
        """
        Measure model stability across several data partitions.

        Cross-validation is run on the same engineered feature
        structure used during training.
        """
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION ANALYSIS")
        print("=" * 60)

        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )

        cv_results = pd.DataFrame(
            {
                "Fold": np.arange(1, len(scores) + 1),
                "R2": scores,
            }
        )
        cv_results.to_csv(
            METRICS_DIR / "cross_validation_scores.csv",
            index=False,
        )

        print(f"Cross-validation R² scores: {scores}")
        print(f"Mean R²: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

        return scores


def load_evaluation_split(
    raw_X: pd.DataFrame,
    raw_y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Reproduce the project's training/test split.

    DataPreprocessor is attempted first so the evaluation split
    remains compatible with the training pipeline. A deterministic
    fallback split is available when preprocessing cannot run.
    """
    preprocessor = DataPreprocessor()

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = preprocessor.run_preprocessing_pipeline()

        print("\nLoaded the train/test split from DataPreprocessor.")

    except Exception as exc:
        print(f"\nDataPreprocessor could not create the split: {exc}")
        print("Using the deterministic fallback split.")

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            raw_X,
            raw_y,
            test_size=0.20,
            random_state=42,
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def run_evaluation_pipeline() -> tuple[
    ModelEvaluator,
    dict[str, float],
]:
    """
    Run the complete evaluation pipeline.
    """
    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)

    evaluator = ModelEvaluator()

    raw_X, raw_y = evaluator.load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_evaluation_split(
        raw_X,
        raw_y,
    )

    # Apply the exact same engineered feature structure
    # to both training and held-out test data.
    X_train_engineered = evaluator.prepare_features(X_train)

    X_test_engineered = evaluator.prepare_features(X_test)

    evaluator.X_train = X_train_engineered
    evaluator.X_test = X_test_engineered
    evaluator.y_train = y_train
    evaluator.y_test = y_test

    metrics, _, _ = evaluator.evaluate(
        X_train_engineered,
        X_test_engineered,
        y_train,
        y_test,
    )

    evaluator.shap_analysis(
        X_train_engineered,
        X_test_engineered,
    )

    evaluator.feature_importance_analysis(
        X_test_engineered,
        y_test,
    )

    all_engineered_features = evaluator.prepare_features(raw_X)

    evaluator.cross_validation(
        all_engineered_features,
        raw_y,
        cv=5,
    )

    evaluator.learning_curve_analysis(
        all_engineered_features,
        raw_y,
    )

    print("\nEvaluation pipeline completed successfully!")

    print("\nThe model's behaviour on unseen data is saved in:")
    print(f"   {UNSEEN_PREDICTIONS_PATH}")

    return evaluator, metrics


if __name__ == "__main__":
    run_evaluation_pipeline()
