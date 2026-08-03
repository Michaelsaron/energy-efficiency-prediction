from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import BASE_FEATURES, FeatureEngineer
from src.utils import load_data

ModelType = Literal["tree", "linear"]


class DataPreprocessor:
    """
    Prepare the energy-efficiency dataset for machine-learning models.

    Compatibility behaviour
    -----------------------
    - Raw data is split before learned preprocessing to prevent data leakage.
    - Feature engineering is applied separately to training and test data.
    - run_preprocessing_pipeline() still returns pandas DataFrames exactly
      as the original implementation did.
    - Tree-based and linear preprocessors are fitted only on training data.
    - Tree-based models receive unscaled numerical features.
    - Linear and distance-based models receive scaled numerical features.
    - Categorical features are one-hot encoded for both model groups.
    """

    def __init__(
        self,
        target_col: str = "Heating_Load",
        test_size: float = 0.2,
        random_state: int = 42,
        apply_feature_engineering: bool = True,
        verbose: bool = True,
    ) -> None:
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.apply_feature_engineering = apply_feature_engineering
        self.verbose = verbose

        self.engineer = FeatureEngineer(verbose=False)

        self.raw_feature_cols = BASE_FEATURES.copy()
        self.feature_cols: list[str] = []

        self.categorical_cols = [
            "Orientation",
            "Glazing_Area_Distribution",
        ]
        self.numerical_cols: list[str] = []

        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

        self.preprocessor_tree: ColumnTransformer | None = None
        self.preprocessor_linear: ColumnTransformer | None = None

        self.is_fitted = False

    def _log(self, message: str) -> None:
        """Print a message when verbose mode is enabled."""
        if self.verbose:
            print(message)

    @staticmethod
    def _create_one_hot_encoder() -> OneHotEncoder:
        """
        Create a OneHotEncoder compatible with both newer and older
        scikit-learn versions.
        """
        common_options = {
            "drop": "first",
            "handle_unknown": "ignore",
        }

        try:
            return OneHotEncoder(
                sparse_output=False,
                **common_options,
            )
        except TypeError:
            # Compatibility with scikit-learn versions older than 1.2.
            return OneHotEncoder(
                sparse=False,
                **common_options,
            )

    def _setup_preprocessors(self, training_data: pd.DataFrame) -> None:
        """
        Create tree and linear preprocessors using the final engineered
        training columns.

        Any engineered numerical features are automatically included.
        """
        missing_categorical = [
            column
            for column in self.categorical_cols
            if column not in training_data.columns
        ]

        if missing_categorical:
            raise ValueError(
                "Categorical columns are missing after feature engineering: "
                + ", ".join(missing_categorical)
            )

        self.numerical_cols = [
            column
            for column in training_data.columns
            if column not in self.categorical_cols
        ]

        self.preprocessor_tree = ColumnTransformer(
            transformers=[
                (
                    "num",
                    "passthrough",
                    self.numerical_cols,
                ),
                (
                    "cat",
                    self._create_one_hot_encoder(),
                    self.categorical_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.preprocessor_linear = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.numerical_cols,
                ),
                (
                    "cat",
                    self._create_one_hot_encoder(),
                    self.categorical_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self._log("✅ Tree and linear preprocessors initialised.")

    def load_and_split(
        self,
        data_path: str | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Load the raw dataset and split it before learned preprocessing.
        """
        df = pd.read_csv(data_path) if data_path else load_data()

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' was not found.")

        missing = [column for column in BASE_FEATURES if column not in df.columns]

        if missing:
            raise ValueError(
                "Dataset missing required input columns: " + ", ".join(missing)
            )

        X = df[BASE_FEATURES].copy()
        y = pd.to_numeric(
            df[self.target_col],
            errors="raise",
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self._log(f"Split: {len(self.X_train)} train / {len(self.X_test)} test")

        return (
            self.X_train.copy(),
            self.X_test.copy(),
            self.y_train.copy(),
            self.y_test.copy(),
        )

    def run_preprocessing_pipeline(
        self,
        data_path: str | None = None,
        **legacy_options,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Split the data, apply deterministic feature engineering, and fit
        both model-specific preprocessors using training data only.

        Additional keyword arguments are accepted for compatibility with
        older calls such as:

            run_preprocessing_pipeline(
                impute=False,
                scale=False,
                cap_outliers=False,
            )

        These options are ignored because scaling and encoding are handled
        by the model-specific preprocessors.
        """
        if legacy_options:
            self._log(
                "Compatibility options received but not applied here: "
                + ", ".join(legacy_options)
            )

        X_train, X_test, y_train, y_test = self.load_and_split(data_path)

        if self.apply_feature_engineering:
            X_train = self.engineer.create_features(X_train.copy())
            X_test = self.engineer.create_features(X_test.copy())

            # Guarantee identical train and test column order.
            X_test = X_test.reindex(columns=X_train.columns)

        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        self.feature_cols = list(self.X_train.columns)

        # Build preprocessors after feature engineering so engineered
        # numerical features are not accidentally dropped.
        self._setup_preprocessors(self.X_train)

        if self.preprocessor_tree is None:
            raise RuntimeError("Tree preprocessor was not created.")

        if self.preprocessor_linear is None:
            raise RuntimeError("Linear preprocessor was not created.")

        # Fit only on training data to prevent leakage.
        self.preprocessor_tree.fit(self.X_train)
        self.preprocessor_linear.fit(self.X_train)

        self.is_fitted = True

        self._log(f"Final model features: {len(self.feature_cols)}")

        return (
            self.X_train.copy(),
            self.X_test.copy(),
            self.y_train.copy(),
            self.y_test.copy(),
        )

    def get_processed_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Return the engineered but not model-transformed datasets.

        This preserves compatibility with the original implementation.
        """
        if not self.is_fitted:
            raise ValueError("Run run_preprocessing_pipeline() first.")

        if (
            self.X_train is None
            or self.X_test is None
            or self.y_train is None
            or self.y_test is None
        ):
            raise RuntimeError("Processed data is unavailable.")

        return (
            self.X_train.copy(),
            self.X_test.copy(),
            self.y_train.copy(),
            self.y_test.copy(),
        )

    def get_preprocessor(
        self,
        model_type: ModelType = "tree",
    ) -> ColumnTransformer:
        """
        Return the fitted preprocessor for a model category.

        Parameters
        ----------
        model_type:
            "tree" for Decision Tree, Random Forest, Extra Trees,
            Gradient Boosting, XGBoost, LightGBM, CatBoost and AdaBoost.

            "linear" for Linear Regression, Ridge, Lasso, ElasticNet,
            Support Vector Regression and other scale-sensitive models.
        """
        if not self.is_fitted:
            raise ValueError("Run run_preprocessing_pipeline() first.")

        if model_type == "tree":
            if self.preprocessor_tree is None:
                raise RuntimeError("Tree preprocessor is unavailable.")
            return self.preprocessor_tree

        if model_type == "linear":
            if self.preprocessor_linear is None:
                raise RuntimeError("Linear preprocessor is unavailable.")
            return self.preprocessor_linear

        raise ValueError("Unknown model_type. Use 'tree' or 'linear'.")

    def _prepare_input_frame(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Make an input DataFrame compatible with the fitted preprocessors.

        The method accepts either:
        - the current seven raw features; or
        - an already engineered DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        prepared = X.copy()

        has_raw_features = all(column in prepared.columns for column in BASE_FEATURES)

        already_engineered = all(
            column in prepared.columns for column in self.feature_cols
        )

        if (
            self.apply_feature_engineering
            and has_raw_features
            and not already_engineered
        ):
            prepared = self.engineer.create_features(prepared[BASE_FEATURES].copy())

        missing = [
            column for column in self.feature_cols if column not in prepared.columns
        ]

        if missing:
            raise ValueError(
                "Input data is missing required processed columns: "
                + ", ".join(missing)
            )

        return prepared.reindex(columns=self.feature_cols)

    def transform_for_model(
        self,
        X: pd.DataFrame,
        model_type: ModelType = "tree",
    ) -> pd.DataFrame:
        """
        Transform data for a particular model category.

        Tree models:
        - numerical features are passed through;
        - categorical features are one-hot encoded.

        Linear models:
        - numerical features are standardised;
        - categorical features are one-hot encoded.
        """
        if not self.is_fitted:
            raise ValueError("Run run_preprocessing_pipeline() first.")

        prepared = self._prepare_input_frame(X)
        preprocessor = self.get_preprocessor(model_type)

        transformed = preprocessor.transform(prepared)
        feature_names = self.get_feature_names_after_encoding(model_type)

        return pd.DataFrame(
            transformed,
            columns=feature_names,
            index=prepared.index,
        )

    def get_feature_names_after_encoding(
        self,
        model_type: ModelType = "tree",
    ) -> list[str]:
        """Return the exact feature names produced by a preprocessor."""
        preprocessor = self.get_preprocessor(model_type)

        try:
            return list(preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older scikit-learn versions.
            feature_names = self.numerical_cols.copy()

            encoder = preprocessor.named_transformers_["cat"]

            try:
                encoded_names = encoder.get_feature_names_out(self.categorical_cols)
            except AttributeError:
                encoded_names = encoder.get_feature_names(self.categorical_cols)

            feature_names.extend(str(name) for name in encoded_names)
            return feature_names

    def get_original_feature_names(self) -> list[str]:
        """
        Return the engineered feature names before scaling and encoding.

        Before fitting, return the original seven input features.
        """
        if not self.is_fitted:
            return self.raw_feature_cols.copy()

        return self.feature_cols.copy()
