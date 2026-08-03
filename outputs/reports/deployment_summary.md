# Deployment Summary

Generated: 2026-08-03 13:09:18

## Deployment Model

- Model name: **CatBoost**
- Estimator class: **CatBoostSklearnAdapter**
- Target: **Heating_Load**
- Raw input count: **7**
- Final model feature count: **8**
- Preprocessing saved inside model: **True**

## Required Deployment Files

| File                                 | Exists   |
|:-------------------------------------|:---------|
| models/best_model.pkl                | True     |
| models/model_metadata.json           | True     |
| models/feature_info.json             | True     |
| models/evaluation_metrics.json       | True     |
| outputs/metrics/model_comparison.csv | True     |

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
