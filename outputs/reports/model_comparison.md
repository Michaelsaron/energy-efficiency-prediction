# Model Comparison Report

Generated: 2026-08-03 13:09:18

## Authoritative Deployment Model

- Model name: **CatBoost**
- Estimator class: **CatBoostSklearnAdapter**
- Test R²: **0.9988**
- Test RMSE: **0.3535**
- Test MAE: **0.2571**
- CV R² mean: **0.9986**
- CV R² standard deviation: **0.0003**
- Selection rule: **Balanced deployment score: 40% Test R2, 15% RMSE, 10% MAE, 20% CV R2 mean, 7.5% CV variability, and 7.5% train-test gap.**

## Model Results

|   Deployment Rank | Algorithm         |   Train R2 |       R2 |     RMSE |      MAE |   CV R2 Mean |   CV R2 Std |   Train-Test R2 Gap |   Training Time (s) |   Deployment Score |
|------------------:|:------------------|-----------:|---------:|---------:|---------:|-------------:|------------:|--------------------:|--------------------:|-------------------:|
|                 1 | CatBoost          |   0.99969  | 0.998801 | 0.353523 | 0.257061 |     0.998588 | 0.000308112 |         0.000888979 |          0.185614   |           99.0909  |
|                 2 | XGBoost           |   0.999671 | 0.998624 | 0.378765 | 0.273794 |     0.998544 | 0.000227351 |         0.00104739  |          0.303892   |           98.5889  |
|                 3 | LightGBM          |   0.998504 | 0.998135 | 0.440922 | 0.324375 |     0.99693  | 0.000695943 |         0.00036875  |          0.731102   |           98.0473  |
|                 4 | Gradient Boosting |   0.998676 | 0.997589 | 0.501295 | 0.364532 |     0.997849 | 0.000369972 |         0.00108716  |          0.226695   |           96.6194  |
|                 5 | Random Forest     |   0.999699 | 0.997694 | 0.490293 | 0.353312 |     0.997271 | 0.000327871 |         0.00200478  |          0.174602   |           95.2942  |
|                 6 | Extra Trees       |   1        | 0.997555 | 0.504852 | 0.350499 |     0.997632 | 0.000138894 |         0.00244528  |          0.111495   |           94.7217  |
|                 7 | Decision Tree     |   0.999778 | 0.996355 | 0.616348 | 0.416857 |     0.996413 | 0.000753812 |         0.00342292  |          0.00426208 |           90.9784  |
|                 8 | SVR               |   0.993372 | 0.992467 | 0.886077 | 0.650766 |     0.981099 | 0.00669606  |         0.00090413  |          0.0849034  |           82.2516  |
|                 9 | AdaBoost          |   0.957558 | 0.952754 | 2.21912  | 1.74133  |     0.958093 | 0.00704193  |         0.00480319  |          0.225286   |           39.0607  |
|                10 | Linear Regression |   0.924318 | 0.919055 | 2.90465  | 2.15366  |     0.922032 | 0.00807077  |         0.00526254  |          0.00807317 |            7.3872  |
|                11 | Lasso Regression  |   0.923515 | 0.918397 | 2.91643  | 2.1167   |     0.921177 | 0.00911513  |         0.00511779  |          0.114598   |            6.46116 |
|                12 | Ridge Regression  |   0.917707 | 0.912372 | 3.02218  | 2.1814   |     0.915087 | 0.0104266   |         0.00533451  |          0.0137372  |            0       |

## Selection Interpretation

The report does not independently choose a model. It reads the same
`model_metadata.json` and `best_model.pkl` used by evaluation, prediction, and
Streamlit. This prevents LightGBM/CatBoost inconsistencies caused by stale files.
