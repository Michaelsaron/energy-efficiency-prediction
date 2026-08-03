from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostSklearnAdapter(BaseEstimator, RegressorMixin):
    """
    Pickle-safe scikit-learn wrapper for CatBoost.

    Keeping this class in a dedicated importable module gives joblib a stable
    class path: src.model_adapters.CatBoostSklearnAdapter.

    This allows the saved model to be loaded by:
    - src.evaluate
    - src.predict
    - app/app.py
    - any other deployment script
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        loss_function: str = "RMSE",
        random_seed: int = 42,
        verbose: bool = False,
        allow_writing_files: bool = False,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.model_ = None

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("CatBoost model has not been fitted.")
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        if self.model_ is None:
            raise AttributeError("CatBoost model has not been fitted.")
        return self.model_.feature_importances_

    def get_feature_importance(self, *args, **kwargs):
        if self.model_ is None:
            raise RuntimeError("CatBoost model has not been fitted.")
        return self.model_.get_feature_importance(*args, **kwargs)
