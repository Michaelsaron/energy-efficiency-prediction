# Final Project Report

Generated: 2026-08-03 13:09:18

## 1. Project Objective

The project predicts a building's heating load from architectural design
characteristics. The result can support energy-efficiency analysis and building
design decisions.

## 2. Dataset

- Rows: **768**
- Columns: **10**
- Target: **Heating_Load**
- Missing values: **0**
- Duplicate rows: **0**

## 3. Leakage-Safe Workflow

The train/test split occurs before learned preprocessing. Feature engineering is
performed separately on training and test data. Imputation and scaling are kept
inside model pipelines and fitted only on training folds.

## 4. Feature Structure

- Raw inputs: **7**
- Final model features: **8**
- Feature engineering enabled: **True**

## 5. Top Model Results

|   Deployment Rank | Algorithm         |       R2 |     RMSE |      MAE |   CV R2 Mean |   CV R2 Std |
|------------------:|:------------------|---------:|---------:|---------:|-------------:|------------:|
|                 1 | CatBoost          | 0.998801 | 0.353523 | 0.257061 |     0.998588 | 0.000308112 |
|                 2 | XGBoost           | 0.998624 | 0.378765 | 0.273794 |     0.998544 | 0.000227351 |
|                 3 | LightGBM          | 0.998135 | 0.440922 | 0.324375 |     0.99693  | 0.000695943 |
|                 4 | Gradient Boosting | 0.997589 | 0.501295 | 0.364532 |     0.997849 | 0.000369972 |
|                 5 | Random Forest     | 0.997694 | 0.490293 | 0.353312 |     0.997271 | 0.000327871 |

## 6. Production Model

- Model name: **CatBoost**
- Estimator class: **CatBoostSklearnAdapter**
- Test R²: **0.9988**
- Test RMSE: **0.3535**
- Test MAE: **0.2571**
- CV R²: **0.9986 ± 0.0003**

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
