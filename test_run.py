"""
Quick project health check.

This script verifies that the main project components are working
before training, evaluation, report generation, or deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

print("=" * 72)
print("ENERGY EFFICIENCY PREDICTION - PROJECT HEALTH CHECK")
print("=" * 72)

# ---------------------------------------------------------------------
# Test 1 - Utilities and folders
# ---------------------------------------------------------------------
print("\n[1] Testing utilities and folders...")

from src.utils import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    load_data,
)

assert PROJECT_ROOT.exists()
assert DATA_DIR.exists()
assert MODELS_DIR.exists()
assert OUTPUTS_DIR.exists()

print(f"✓ Project Root : {PROJECT_ROOT}")
print(f"✓ Data Folder  : {DATA_DIR}")
print(f"✓ Models Folder: {MODELS_DIR}")
print(f"✓ Outputs      : {OUTPUTS_DIR}")

# ---------------------------------------------------------------------
# Test 2 - Dataset
# ---------------------------------------------------------------------
print("\n[2] Testing dataset loading...")

df = load_data(verbose=True)

required_columns = {
    "Relative_Compactness",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
    "Heating_Load",
}

missing_columns = required_columns.difference(df.columns)

assert not df.empty, "Dataset is empty."
assert not missing_columns, (
    "Dataset is missing required columns: "
    + ", ".join(sorted(missing_columns))
)

print(f"✓ Dataset shape : {df.shape}")
print(f"✓ Columns       : {len(df.columns)}")
print(f"✓ Missing cells : {int(df.isna().sum().sum())}")
print(f"✓ Duplicate rows: {int(df.duplicated().sum())}")

# ---------------------------------------------------------------------
# Test 3 - Preprocessing
# ---------------------------------------------------------------------
print("\n[3] Testing leakage-safe preprocessing pipeline...")

from src.preprocessing import DataPreprocessor

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
assert X_train.shape[1] == 8
assert X_test.shape[1] == 8

print(f"✓ Train shape : {X_train.shape}")
print(f"✓ Test shape  : {X_test.shape}")
print("✓ Train and test indices are disjoint")
print("✓ Final feature count: 8")

# ---------------------------------------------------------------------
# Test 4 - Feature engineering
# ---------------------------------------------------------------------
print("\n[4] Checking engineered features...")

expected_engineered = {"Compactness_Height"}

missing_engineered = expected_engineered.difference(X_train.columns)

assert not missing_engineered, (
    "Missing engineered features: "
    + ", ".join(sorted(missing_engineered))
)

assert X_train.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

print("✓ Engineered features:")
for feature in sorted(expected_engineered):
    print(f"  - {feature}")

print("✓ No missing values in engineered train/test data")

# ---------------------------------------------------------------------
# Test 5 - Model-specific preprocessing
# ---------------------------------------------------------------------
print("\n[5] Testing tree and linear preprocessors...")

tree_train = preprocessor.transform_for_model(
    X_train,
    model_type="tree",
)

linear_train = preprocessor.transform_for_model(
    X_train,
    model_type="linear",
)

assert tree_train.shape[0] == len(X_train)
assert linear_train.shape[0] == len(X_train)
assert tree_train.shape[1] == linear_train.shape[1]

print(f"✓ Tree-transformed shape  : {tree_train.shape}")
print(f"✓ Linear-transformed shape: {linear_train.shape}")

# ---------------------------------------------------------------------
# Test 6 - JSON metadata utilities
# ---------------------------------------------------------------------
print("\n[6] Testing JSON read/write...")

from src.utils import load_json, save_json

test_json_path = MODELS_DIR / "health_check_test.json"

save_json(
    {"status": "ok"},
    test_json_path.name,
    verbose=False,
)

metadata = load_json(test_json_path.name, required=True)

assert metadata["status"] == "ok"

test_json_path.unlink(missing_ok=True)

print("✓ JSON read/write successful")
print("✓ Temporary test file removed")

# ---------------------------------------------------------------------
# Test 7 - Optional saved model consistency
# ---------------------------------------------------------------------
print("\n[7] Checking saved model artefacts...")

required_model_files = [
    MODELS_DIR / "best_model.pkl",
    MODELS_DIR / "feature_info.json",
    MODELS_DIR / "model_metadata.json",
]

missing_model_files = [
    path.name for path in required_model_files if not path.exists()
]

if missing_model_files:
    print(
        "⚠ Saved model artefacts are not complete yet: "
        + ", ".join(missing_model_files)
    )
    print("  Run: python -m src.train")
else:
    from src.predict import EnergyPredictor

    predictor = EnergyPredictor()

    sample = {
        "Relative_Compactness": 0.86,
        "Wall_Area": 294.0,
        "Roof_Area": 147.0,
        "Overall_Height": 7.0,
        "Orientation": 2,
        "Glazing_Area": 0.10,
        "Glazing_Area_Distribution": 1,
    }

    prediction = predictor.predict_one(sample)

    assert isinstance(prediction, float)

    print(f"✓ Saved model loaded successfully")
    print(f"✓ Sample prediction: {prediction:.4f}")

# ---------------------------------------------------------------------
# Test Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 72)
print("PROJECT HEALTH CHECK PASSED")
print("=" * 72)

print("\nRecommended run order:")
print("1. python -m pytest tests -v")
print("2. python test_run.py")
print("3. python -m src.train")
print("4. python -m src.evaluate")
print("5. python -m src.report_generator")
print("6. streamlit run app/app.py")
