# src/report_generator.py
"""
Generate consistent Markdown reports for the Energy Efficiency Prediction project.

Generated files
---------------
outputs/reports/
├── data_understanding.md
├── eda_summary.md
├── feature_engineering.md
├── model_comparison.md
├── model_evaluation.md
├── deployment_summary.md
└── final_project_report.md

Run:
    python -m src.report_generator

Recommended workflow:
    python -m src.train
    python -m src.evaluate
    python -m src.report_generator
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    estimator_name,
    load_data,
    load_json,
    load_model,
)


TARGET_COLUMN = "Heating_Load"


def _timestamp() -> str:
    """Return a readable report-generation timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_report(file_name: str, content: str) -> Path:
    """
    Write a Markdown report to outputs/reports.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / file_name
    output_path.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )

    print(f"Generated: {output_path}")
    return output_path


def _markdown_table(
    dataframe: pd.DataFrame | None,
    *,
    max_rows: int | None = None,
) -> str:
    """
    Convert a DataFrame to a Markdown table.
    """
    if dataframe is None or dataframe.empty:
        return "_No data available._"

    output = dataframe.copy()

    if max_rows is not None:
        output = output.head(max_rows)

    return output.to_markdown(index=False)


def _format_number(
    value: Any,
    decimals: int = 4,
) -> str:
    """
    Safely format numerical values.
    """
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "Not available"


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """
    Read a CSV file if it exists.
    """
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: Could not read {path}: {exc}")
        return None


def _list_figure_files(
    keywords: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Return generated figure filenames.

    When keywords are supplied, include only matching files.
    """
    if not FIGURES_DIR.exists():
        return []

    files = sorted(
        path.name
        for path in FIGURES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    )

    if keywords is None:
        return files

    return [
        file_name
        for file_name in files
        if any(
            keyword.lower() in file_name.lower()
            for keyword in keywords
        )
    ]


def _figure_markdown(files: list[str]) -> str:
    """
    Convert figure filenames into a Markdown list.
    """
    if not files:
        return "_No matching figures are currently available._"

    return "\n".join(
        f"- `outputs/figures/{file_name}`"
        for file_name in files
    )


def _normalise_comparison_columns(
    results_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Support older and newer model-comparison column names.
    """
    if results_df is None:
        return None

    dataframe = results_df.copy()

    aliases = {
        "Algorithm Name": "Algorithm",
        "R2 Score": "R2",
        "Test R2": "R2",
        "Test RMSE": "RMSE",
        "Test MAE": "MAE",
        "Cross Validation R2 Mean": "CV R2 Mean",
        "Cross Validation R2 Std": "CV R2 Std",
        "Train-Test Gap": "Train-Test R2 Gap",
        "Overfit Gap": "Train-Test R2 Gap",
    }

    rename_map = {
        old_name: new_name
        for old_name, new_name in aliases.items()
        if old_name in dataframe.columns
        and new_name not in dataframe.columns
    }

    if rename_map:
        dataframe = dataframe.rename(columns=rename_map)

    return dataframe


def validate_deployment_consistency() -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    """
    Validate that the saved model and metadata describe the same estimator.

    Raises
    ------
    RuntimeError
        If best_model.pkl and model_metadata.json disagree.
    """
    model = load_model("best_model.pkl")
    metadata = load_json(
        "model_metadata.json",
        required=True,
    )
    feature_info = load_json(
        "feature_info.json",
        required=True,
    )

    actual_estimator = estimator_name(model)
    expected_estimator = metadata.get("estimator_class")

    if expected_estimator and expected_estimator != actual_estimator:
        raise RuntimeError(
            "Deployment files are inconsistent.\n"
            f"model_metadata.json says: {expected_estimator}\n"
            f"best_model.pkl contains: {actual_estimator}\n"
            "Delete the old model artefacts and rerun:\n"
            "  python -m src.train"
        )

    metadata_model_name = metadata.get("best_model")

    if not metadata_model_name:
        raise RuntimeError(
            "model_metadata.json does not contain 'best_model'. "
            "Rerun: python -m src.train"
        )

    required_feature_keys = {
        "raw_feature_names",
        "feature_names",
        "n_raw_features",
        "n_features",
        "target",
    }

    missing_keys = sorted(
        required_feature_keys - set(feature_info)
    )

    if missing_keys:
        raise RuntimeError(
            "feature_info.json is incomplete. Missing keys: "
            + ", ".join(missing_keys)
            + "\nRerun: python -m src.train"
        )

    if len(feature_info["raw_feature_names"]) != feature_info["n_raw_features"]:
        raise RuntimeError(
            "feature_info.json has inconsistent raw feature counts."
        )

    if len(feature_info["feature_names"]) != feature_info["n_features"]:
        raise RuntimeError(
            "feature_info.json has inconsistent final feature counts."
        )

    return model, metadata, feature_info


def generate_data_understanding_report(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Generate the dataset overview report.
    """
    data_types = (
        dataframe.dtypes.astype(str)
        .rename("Data Type")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    missing_values = (
        dataframe.isna()
        .sum()
        .rename("Missing Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    unique_values = (
        dataframe.nunique(dropna=False)
        .rename("Unique Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    column_summary = (
        data_types
        .merge(missing_values, on="Column", how="left")
        .merge(unique_values, on="Column", how="left")
    )

    descriptive_statistics = (
        dataframe.describe(include="all")
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    content = f"""
# Data Understanding Report

Generated: {_timestamp()}

## Dataset Overview

- Rows: **{len(dataframe):,}**
- Columns: **{len(dataframe.columns)}**
- Target variable: **{TARGET_COLUMN}**
- Duplicate rows: **{int(dataframe.duplicated().sum())}**
- Total missing values: **{int(dataframe.isna().sum().sum())}**

## Column Summary

{_markdown_table(column_summary)}

## Descriptive Statistics

{_markdown_table(descriptive_statistics)}
"""

    return _write_report(
        "data_understanding.md",
        content,
    )


def generate_eda_summary_report(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Generate correlation, skewness, and outlier summaries.
    """
    numerical = dataframe.select_dtypes(
        include=[np.number]
    )

    target_correlations = pd.DataFrame()

    if TARGET_COLUMN in numerical.columns:
        target_correlations = (
            numerical.corr()[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(
                key=lambda series: series.abs(),
                ascending=False,
            )
            .rename("Correlation")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )

    skewness = (
        numerical.skew()
        .sort_values(
            key=lambda series: series.abs(),
            ascending=False,
        )
        .rename("Skewness")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    outlier_rows: list[dict[str, Any]] = []

    for column in numerical.columns:
        q1 = numerical[column].quantile(0.25)
        q3 = numerical[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (numerical[column] < lower_bound)
            | (numerical[column] > upper_bound)
        )

        outlier_rows.append(
            {
                "Feature": column,
                "Outlier Count": int(outlier_mask.sum()),
                "Outlier Percentage": round(
                    float(outlier_mask.mean() * 100),
                    2,
                ),
                "Lower IQR Bound": round(
                    float(lower_bound),
                    4,
                ),
                "Upper IQR Bound": round(
                    float(upper_bound),
                    4,
                ),
            }
        )

    outlier_report = (
        pd.DataFrame(outlier_rows)
        .sort_values(
            "Outlier Count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    figures = _list_figure_files(
        (
            "distribution",
            "boxplot",
            "correlation",
            "relationship",
            "scatter",
            "pairplot",
        )
    )

    content = f"""
# Exploratory Data Analysis Summary

Generated: {_timestamp()}

## Correlation with Heating Load

{_markdown_table(target_correlations)}

## Feature Skewness

{_markdown_table(skewness)}

## IQR Outlier Report

The IQR analysis identifies unusual values for review. Values are not
automatically removed because they may represent valid building configurations.

{_markdown_table(outlier_report)}

## Generated EDA Figures

{_figure_markdown(figures)}

## Main Interpretation

- Features with larger absolute correlations have stronger linear relationships
  with heating load.
- Strong skewness may indicate non-normal distributions or discrete design
  categories.
- Reported outliers should be evaluated in context before capping or removal.
- Tree and boosting models can capture nonlinear relationships that correlation
  alone may not reveal.
"""

    return _write_report(
        "eda_summary.md",
        content,
    )


def generate_feature_engineering_report(
    feature_info: dict[str, Any],
) -> Path:
    """
    Generate raw and engineered feature documentation.
    """
    raw_features = list(
        feature_info.get("raw_feature_names", [])
    )
    final_features = list(
        feature_info.get("feature_names", [])
    )

    engineered_features = [
        feature
        for feature in final_features
        if feature not in raw_features
    ]

    raw_table = pd.DataFrame(
        {"Raw Feature": raw_features}
    )

    engineered_table = pd.DataFrame(
        {"Engineered Feature": engineered_features}
    )

    content = f"""
# Feature Engineering Report

Generated: {_timestamp()}

## Feature Counts

- Raw input features: **{feature_info["n_raw_features"]}**
- Final model features: **{feature_info["n_features"]}**
- Engineered features: **{len(engineered_features)}**
- Target: **{feature_info["target"]}**
- Feature engineering enabled: **{feature_info.get("feature_engineering", True)}**

## Raw Input Features

{_markdown_table(raw_table)}

## Engineered Features

{_markdown_table(engineered_table)}

## Leakage Prevention

The dataset is split before learned preprocessing. Deterministic feature
engineering is applied separately to training and test data.

Imputation and scaling are fitted only through each model's training pipeline,
so test-set statistics are not used during training.

## Deployment Feature Order

The exact ordered model feature list is stored in:

```text
models/feature_info.json
```

The prediction module uses this order before calling `best_model.pkl`.
"""

    return _write_report(
        "feature_engineering.md",
        content,
    )


def generate_model_comparison_report(
    results_df: pd.DataFrame | None,
    metadata: dict[str, Any],
) -> Path:
    """
    Generate the model comparison report.
    """
    results_df = _normalise_comparison_columns(
        results_df
    )

    if results_df is None or results_df.empty:
        comparison_table = (
            "_No comparison results found. "
            "Run `python -m src.train` first._"
        )
    else:
        preferred_columns = [
            "Deployment Rank",
            "Algorithm",
            "Train R2",
            "R2",
            "RMSE",
            "MAE",
            "CV R2 Mean",
            "CV R2 Std",
            "Train-Test R2 Gap",
            "Training Time (s)",
            "Deployment Score",
        ]

        selected_columns = [
            column
            for column in preferred_columns
            if column in results_df.columns
        ]

        comparison_table = _markdown_table(
            results_df[selected_columns]
        )

    content = f"""
# Model Comparison Report

Generated: {_timestamp()}

## Authoritative Deployment Model

- Model name: **{metadata["best_model"]}**
- Estimator class: **{metadata.get("estimator_class", "Not available")}**
- Test R²: **{_format_number(metadata.get("best_r2"))}**
- Test RMSE: **{_format_number(metadata.get("best_rmse"))}**
- Test MAE: **{_format_number(metadata.get("best_mae"))}**
- CV R² mean: **{_format_number(metadata.get("cv_r2_mean"))}**
- CV R² standard deviation: **{_format_number(metadata.get("cv_r2_std"))}**
- Selection rule: **{metadata.get("selection_rule", "Not available")}**

## Model Results

{comparison_table}

## Selection Interpretation

The report does not independently choose a model. It reads the same
`model_metadata.json` and `best_model.pkl` used by evaluation, prediction, and
Streamlit. This prevents LightGBM/CatBoost inconsistencies caused by stale files.
"""

    return _write_report(
        "model_comparison.md",
        content,
    )


def generate_model_evaluation_report(
    evaluation_metrics: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> Path:
    """
    Generate best-model evaluation documentation.
    """
    evaluation_metrics = evaluation_metrics or {}

    metric_rows: list[dict[str, Any]] = []

    for metric_name, value in evaluation_metrics.items():
        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            display_value = _format_number(value)
        else:
            display_value = str(value)

        metric_rows.append(
            {
                "Metric": metric_name,
                "Value": display_value,
            }
        )

    metrics_table = pd.DataFrame(metric_rows)

    figures = _list_figure_files(
        (
            "actual",
            "predicted",
            "residual",
            "qq",
            "importance",
            "learning",
            "shap",
        )
    )

    content = f"""
# Model Evaluation Report

Generated: {_timestamp()}

## Evaluated Deployment Model

- Model name: **{metadata["best_model"]}**
- Estimator class: **{metadata.get("estimator_class", "Not available")}**

## Evaluation Metrics

{_markdown_table(metrics_table)}

## Diagnostic Figures

{_figure_markdown(figures)}

## Evaluation Guidance

The deployment model should be evaluated using:

- train and test MAE;
- train and test RMSE;
- train and test R²;
- train-test R² gap;
- shuffled cross-validation;
- residual behaviour;
- actual-versus-predicted agreement;
- feature importance or SHAP explanations.

A relatively small train-test gap and stable cross-validation scores indicate
stronger generalisation.
"""

    return _write_report(
        "model_evaluation.md",
        content,
    )


def generate_deployment_summary_report(
    metadata: dict[str, Any],
    feature_info: dict[str, Any],
) -> Path:
    """
    Generate deployment-file and inference-flow documentation.
    """
    required_files = [
        MODELS_DIR / "best_model.pkl",
        MODELS_DIR / "model_metadata.json",
        MODELS_DIR / "feature_info.json",
        MODELS_DIR / "evaluation_metrics.json",
        METRICS_DIR / "model_comparison.csv",
    ]

    file_status = pd.DataFrame(
        [
            {
                "File": str(
                    file_path.relative_to(
                        MODELS_DIR.parent
                    )
                ),
                "Exists": file_path.exists(),
            }
            for file_path in required_files
        ]
    )

    content = f"""
# Deployment Summary

Generated: {_timestamp()}

## Deployment Model

- Model name: **{metadata["best_model"]}**
- Estimator class: **{metadata.get("estimator_class", "Not available")}**
- Target: **{feature_info["target"]}**
- Raw input count: **{feature_info["n_raw_features"]}**
- Final model feature count: **{feature_info["n_features"]}**
- Preprocessing saved inside model: **{feature_info.get("preprocessing_saved_inside_model", True)}**

## Required Deployment Files

{_markdown_table(file_status)}

## Prediction Flow

```text
Streamlit form or uploaded CSV
              ↓
src/predict.py
              ↓
raw input validation
              ↓
src/feature_engineering.py
              ↓
feature ordering from feature_info.json
              ↓
best_model.pkl
              ↓
Heating Load prediction
```

## Main Commands

```bash
python -m src.train
python -m src.evaluate
python -m src.report_generator
streamlit run app/app.py
```

## Consistency Protection

Before generating reports, this module verifies that:

- `best_model.pkl` can be loaded;
- `model_metadata.json` identifies the same estimator class;
- `feature_info.json` contains complete feature metadata;
- raw and final feature counts match their saved lists.

If these checks fail, reports are not generated from stale or conflicting files.
"""

    return _write_report(
        "deployment_summary.md",
        content,
    )


def generate_final_project_report(
    dataframe: pd.DataFrame,
    results_df: pd.DataFrame | None,
    metadata: dict[str, Any],
    feature_info: dict[str, Any],
    evaluation_metrics: dict[str, Any] | None,
) -> Path:
    """
    Generate a concise final project report.
    """
    results_df = _normalise_comparison_columns(
        results_df
    )

    if results_df is None or results_df.empty:
        top_models_table = "_No model results available._"
    else:
        preferred_columns = [
            "Deployment Rank",
            "Algorithm",
            "R2",
            "RMSE",
            "MAE",
            "CV R2 Mean",
            "CV R2 Std",
        ]

        selected_columns = [
            column
            for column in preferred_columns
            if column in results_df.columns
        ]

        top_models_table = _markdown_table(
            results_df[selected_columns],
            max_rows=5,
        )

    evaluation_metrics = evaluation_metrics or {}

    test_r2 = evaluation_metrics.get(
        "test_r2",
        metadata.get("best_r2"),
    )
    test_rmse = evaluation_metrics.get(
        "test_rmse",
        metadata.get("best_rmse"),
    )
    test_mae = evaluation_metrics.get(
        "test_mae",
        metadata.get("best_mae"),
    )
    cv_mean = evaluation_metrics.get(
        "cv_r2_mean",
        metadata.get("cv_r2_mean"),
    )
    cv_std = evaluation_metrics.get(
        "cv_r2_std",
        metadata.get("cv_r2_std"),
    )

    content = f"""
# Final Project Report

Generated: {_timestamp()}

## 1. Project Objective

The project predicts a building's heating load from architectural design
characteristics. The result can support energy-efficiency analysis and building
design decisions.

## 2. Dataset

- Rows: **{len(dataframe):,}**
- Columns: **{len(dataframe.columns)}**
- Target: **{feature_info["target"]}**
- Missing values: **{int(dataframe.isna().sum().sum())}**
- Duplicate rows: **{int(dataframe.duplicated().sum())}**

## 3. Leakage-Safe Workflow

The train/test split occurs before learned preprocessing. Feature engineering is
performed separately on training and test data. Imputation and scaling are kept
inside model pipelines and fitted only on training folds.

## 4. Feature Structure

- Raw inputs: **{feature_info["n_raw_features"]}**
- Final model features: **{feature_info["n_features"]}**
- Feature engineering enabled: **{feature_info.get("feature_engineering", True)}**

## 5. Top Model Results

{top_models_table}

## 6. Production Model

- Model name: **{metadata["best_model"]}**
- Estimator class: **{metadata.get("estimator_class", "Not available")}**
- Test R²: **{_format_number(test_r2)}**
- Test RMSE: **{_format_number(test_rmse)}**
- Test MAE: **{_format_number(test_mae)}**
- CV R²: **{_format_number(cv_mean)} ± {_format_number(cv_std)}**

## 7. Evaluation

The selected model is reviewed through holdout metrics, shuffled
cross-validation, residual diagnostics, actual-versus-predicted plots, and
feature-importance or SHAP analysis.

## 8. Deployment

The complete fitted production pipeline is stored in:

```text
models/best_model.pkl
```

The same metadata is used by:

- `src/predict.py`;
- `src/evaluate.py`;
- `src/report_generator.py`;
- the Streamlit application.

## 9. Main Challenges

- preventing data leakage;
- ensuring consistent engineered-feature order;
- avoiding double feature engineering;
- maintaining package-version compatibility;
- ensuring the notebook, saved model, metadata, reports, and Streamlit display
  the same production model.

## 10. Report Quality and Reproducibility

- performance values come from held-out test data or cross-validation;
- figures are saved with descriptive names in `outputs/figures/`;
- limitations and possible model errors are reported honestly;
- the workflow can be reproduced from the README instructions;
- code logic is summarised rather than copied into the report.

## 11. Future Improvements

- hyperparameter optimisation;
- automated model monitoring;
- drift detection;
- uncertainty intervals;
- continuous integration tests;
- cloud deployment;
- additional building and climate variables.
"""

    return _write_report(
        "final_project_report.md",
        content,
    )


def generate_all_reports() -> list[Path]:
    """
    Validate deployment artefacts and generate all Markdown reports.
    """
    print("\n" + "=" * 72)
    print("GENERATING ENERGY EFFICIENCY PROJECT REPORTS")
    print("=" * 72)

    _, metadata, feature_info = (
        validate_deployment_consistency()
    )

    dataframe = load_data(verbose=False)

    comparison_results = _read_csv_if_exists(
        METRICS_DIR / "model_comparison.csv"
    )

    evaluation_metrics = load_json(
        "evaluation_metrics.json",
        required=False,
    )

    generated_reports = [
        generate_data_understanding_report(
            dataframe
        ),
        generate_eda_summary_report(
            dataframe
        ),
        generate_feature_engineering_report(
            feature_info
        ),
        generate_model_comparison_report(
            comparison_results,
            metadata,
        ),
        generate_model_evaluation_report(
            evaluation_metrics,
            metadata,
        ),
        generate_deployment_summary_report(
            metadata,
            feature_info,
        ),
        generate_final_project_report(
            dataframe,
            comparison_results,
            metadata,
            feature_info,
            evaluation_metrics,
        ),
    ]

    print("\nAll reports generated successfully.")
    print(
        f"Authoritative deployment model: "
        f"{metadata['best_model']} "
        f"({metadata.get('estimator_class', 'Unknown class')})"
    )

    return generated_reports


if __name__ == "__main__":
    generate_all_reports()
