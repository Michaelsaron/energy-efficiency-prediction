# Feature Engineering Report

Generated: 2026-08-03 13:09:18

## Feature Counts

- Raw input features: **7**
- Final model features: **8**
- Engineered features: **1**
- Target: **Heating_Load**
- Feature engineering enabled: **True**

## Raw Input Features

| Raw Feature               |
|:--------------------------|
| Relative_Compactness      |
| Wall_Area                 |
| Roof_Area                 |
| Overall_Height            |
| Orientation               |
| Glazing_Area              |
| Glazing_Area_Distribution |

## Engineered Features

| Engineered Feature   |
|:---------------------|
| Compactness_Height   |

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
