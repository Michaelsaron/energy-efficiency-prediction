# Energy Efficiency Prediction

A complete machine-learning regression project that predicts the **Heating Load** of residential buildings from architectural characteristics.

The project includes leakage-safe preprocessing, limited feature engineering, exploratory data analysis, comparison of 12 regression algorithms, held-out test evaluation, cross-validation, SHAP explainability, learning curves, residual analysis, MySQL authentication and prediction history, batch prediction, PDF reports and a Streamlit application.

## Business problem

Building design decisions affect future heating-energy demand. Estimating heating load before construction can help designers compare alternatives and make more energy-efficient decisions.

## Project objective

Predict the continuous target variable `Heating_Load` using seven building-design inputs.

This is a **regression problem** because the target is a continuous numerical value.

## Dataset

The project uses the UCI Energy Efficiency dataset stored locally as:

```text
data/energy_efficiency_data.csv
```

Expected dataset size:

- 768 rows
- 10 columns
- 7 production input features
- 2 target columns
- `Surface_Area` retained in the source dataset for EDA only

Production target:

```text
Heating_Load
```

Raw input features:

1. `Relative_Compactness`
2. `Wall_Area`
3. `Roof_Area`
4. `Overall_Height`
5. `Orientation`
6. `Glazing_Area`
7. `Glazing_Area_Distribution`

The project predicts `Heating_Load`. `Cooling_Load` is not used as an input.

## Leakage-safe workflow

```text
Dataset
  ↓
Data validation
  ↓
Train/test split
  ↓
Feature engineering applied separately
  ↓
Model-specific preprocessing fitted on training data only
  ↓
Train and compare 12 regression models
  ↓
Evaluate on held-out unseen test data
  ↓
Cross-validation and learning curve
  ↓
Save the selected production model
  ↓
Generate reports and visualisations
  ↓
Serve predictions through Streamlit
```

The train/test split occurs before learned preprocessing. Scaling, encoding and imputation are fitted only on training data or training folds.

## Feature engineering

One concise engineered feature is used:

| Engineered feature | Meaning |
|---|---|
| `Compactness_Height` | Relative compactness multiplied by overall height |

Final model input:

```text
7 raw features + 1 engineered feature = 8 features
```

`Surface_Area` is excluded from model training because of multicollinearity,
but remains in the raw dataset for EDA and traceability.

Keeping feature engineering limited reduces unnecessary complexity and helps control overfitting.

## Preprocessing

The preprocessing system:

- splits raw data before learned transformations;
- applies feature engineering separately to training and test data;
- one-hot encodes `Orientation` and `Glazing_Area_Distribution`;
- scales numerical features for linear and distance-based models;
- leaves numerical features unscaled for tree-based models;
- uses `handle_unknown="ignore"` for unseen categorical values;
- fits all preprocessors on training data only.

## Regression algorithms

The training pipeline compares:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Support Vector Regressor
8. XGBoost
9. Extra Trees Regressor
10. AdaBoost Regressor
11. CatBoost
12. LightGBM

## Evaluation metrics

Every model is compared using:

| Metric | Meaning | Better value |
|---|---|---|
| MAE | Average absolute prediction error | Lower |
| MSE | Average squared prediction error | Lower |
| RMSE | Error in the target's original scale, with larger errors penalised more | Lower |
| R² | Proportion of target variation explained by the model | Higher |
| CV R² mean | Average R² across cross-validation folds | Higher |
| CV R² standard deviation | Variation across folds | Lower |
| Train-test R² gap | Difference between training and held-out performance | Closer to zero |
| Training time | Time required to fit the model | Lower, when accuracy is similar |

The production model is selected using a balanced deployment score:

1. held-out Test R²;
2. RMSE and MAE;
3. cross-validation R²;
4. cross-validation variability;
5. train-test R² gap.

This prevents selection from relying on R² alone.

The actual winner is determined every time `python -m src.train` is run. Do not hardcode CatBoost, LightGBM or another model as the winner.

## Unseen-data evaluation

After training, `src.evaluate` evaluates the selected model using the held-out test partition that was not used to fit the model.

Important outputs:

```text
models/evaluation_metrics.json
outputs/metrics/unseen_test_predictions.csv
outputs/metrics/learning_curve.csv
```

`unseen_test_predictions.csv` contains:

- actual value;
- predicted value;
- residual;
- absolute error;
- squared error.

The Streamlit Model Information and Model Insights pages read these latest files automatically.

## Exploratory data analysis

The project includes:

- histograms;
- boxplots;
- scatter plots;
- correlation heatmap;
- distribution plots;
- skewness analysis;
- outlier summaries;
- target correlations;
- feature relationships.

EDA results and generated figures are stored under:

```text
outputs/metrics/
outputs/figures/
```

## Streamlit application

Pages:

- Home
- Project Description
- Dataset Information
- Model Information
- Prediction
- Model Insights
- Batch Prediction
- Prediction History
- Model Comparison
- Team Members

Features:

- single prediction;
- prediction result and explanation;
- batch CSV prediction;
- SHAP local explanation;
- global feature importance;
- unseen-data residual analysis;
- actual-versus-predicted chart;
- learning curve;
- downloadable PDF prediction report;
- MySQL authentication;
- MySQL prediction history;
- light and dark themes.

The UI remains unchanged when the model is retrained. Content and visualisations are loaded from the latest saved artefacts.

## MySQL configuration

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
[mysql]
host = "localhost"
port = 3306
database = "energy_efficiency"
user = "root"
password = "YOUR_MYSQL_PASSWORD"
```

Create the database once:

```sql
CREATE DATABASE energy_efficiency
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```

The application creates the `users` and `predictions` tables automatically.

Do not commit or publicly share your real database password.

## Project structure

```text
energy-efficiency-prediction/
├── .streamlit/
│   └── secrets.toml
├── app/
│   └── app.py
├── assets/
│   └── team/
├── data/
│   └── energy_efficiency_data.csv
├── models/
│   ├── best_model.pkl
│   ├── evaluation_metrics.json
│   ├── feature_info.json
│   └── model_metadata.json
├── notebooks/
│   ├── eda.ipynb
│   └── model_comparison.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── src/
│   ├── __init__.py
│   ├── auth.py
│   ├── database.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── report_generator.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Generated artefacts

Training generates or overwrites:

```text
models/best_model.pkl
models/feature_info.json
models/model_metadata.json
outputs/metrics/model_comparison.csv
outputs/metrics/deployment_ranking.csv
```

Evaluation generates or overwrites:

```text
models/evaluation_metrics.json
outputs/metrics/unseen_test_predictions.csv
outputs/metrics/learning_curve.csv
outputs/figures/learning_curve.png
outputs/figures/shap_summary.png
outputs/figures/shap_bar.png
```

Additional residual and actual-versus-predicted figures are also written to `outputs/figures/`.

Report generation creates:

```text
outputs/reports/data_understanding.md
outputs/reports/eda_summary.md
outputs/reports/feature_engineering.md
outputs/reports/model_comparison.md
outputs/reports/model_evaluation.md
outputs/reports/deployment_summary.md
outputs/reports/final_project_report.md
```

## Installation

From the project root:

```bash
python -m pip install -r requirements.txt
```

The requirements must include at least:

```text
pandas
numpy
scikit-learn
joblib
matplotlib
plotly
streamlit
shap
reportlab
bcrypt
mysql-connector-python
xgboost
lightgbm
catboost
pytest
```

## Run the project

Run commands from the main project folder.

### 1. Run tests

```bash
python -m pytest tests -v
```

### 2. Train and compare models

```bash
python -m src.train
```

### 3. Evaluate the selected model on unseen data

```bash
python -m src.evaluate
```

### 4. Generate technical reports

```bash
python -m src.report_generator
```

### 5. Start Streamlit

```bash
streamlit run app/app.py
```

Complete order:

```bash
python -m pytest tests -v
python -m src.train
python -m src.evaluate
python -m src.report_generator
streamlit run app/app.py
```

## When to rerun commands

After changing preprocessing, feature engineering or model settings:

```bash
python -m pytest tests -v
python -m src.train
python -m src.evaluate
python -m src.report_generator
```

After changing only the Streamlit UI:

```bash
streamlit run app/app.py
```

After retraining while Streamlit is running, refresh the browser. The app reads the latest model metadata and metric files.

## Tests

`tests/test_preprocessing.py` checks:

- the correct seven raw features;
- the `Compactness_Height` engineered feature;
- the total of 8 model features;
- safe division and feature values;
- no mutation of the original input;
- disjoint training and test indices;
- deterministic splitting;
- fitted tree and linear preprocessors;
- finite transformed values;
- clear errors for missing features;
- protection against using processed data before fitting.

## Reports

Generate the complete Markdown technical report set with:

```bash
python -m src.report_generator
```

The final report covers:

1. Introduction
2. Business problem
3. Dataset description
4. Data preprocessing
5. Exploratory data analysis
6. Regression algorithms
7. Model evaluation
8. Model comparison
9. Best-model selection
10. Web application
11. Conclusion
12. Future improvements

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Matplotlib
- Plotly
- Streamlit
- MySQL
- mysql-connector-python
- bcrypt
- ReportLab
- Joblib
- Pytest

## Team

**Group 7 — Energy Efficiency Prediction Project**

## Important security note

The MySQL password is stored in `.streamlit/secrets.toml`. Keep that file private and do not upload it to public repositories.

## Licence

This project was developed for educational and research purposes.




```text
outputs/reports/technical_report_checklist.md
```
