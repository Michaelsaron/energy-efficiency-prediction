# Model Evaluation Report

Generated: 2026-08-03 13:09:18

## Evaluated Deployment Model

- Model name: **CatBoost**
- Estimator class: **CatBoostSklearnAdapter**

## Evaluation Metrics

| Metric            |   Value |
|:------------------|--------:|
| Train MAE         |  0.1298 |
| Train MSE         |  0.0313 |
| Train RMSE        |  0.1769 |
| Train R2          |  0.9997 |
| Test MAE          |  0.2571 |
| Test MSE          |  0.125  |
| Test RMSE         |  0.3535 |
| Test R2           |  0.9988 |
| R2 Train-Test Gap |  0.0009 |

## Diagnostic Figures

- `outputs/figures/best_model_actual_vs_predicted.png`
- `outputs/figures/best_model_feature_importance.png`
- `outputs/figures/best_model_qq.png`
- `outputs/figures/best_model_residual_distribution.png`
- `outputs/figures/best_model_residuals.png`
- `outputs/figures/learning_curve.png`
- `outputs/figures/unseen_test_residual_analysis.png`

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
