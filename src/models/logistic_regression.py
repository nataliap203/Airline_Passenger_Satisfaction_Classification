import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.0, 1.0],
}


def build_logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    C: float = 1.0,
    l1_ratio: float = 0.0,
    max_iter: int = 5000,
    random_state: int = 42,
) -> Pipeline:

    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver="saga",
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_logistic_regression(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:

    pipeline.fit(X_train, y_train)
    return pipeline


def tune_hyperparameters(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> GridSearchCV:

    if param_grid is None:
        param_grid = PARAM_GRID

    pipeline = build_logistic_regression_pipeline(preprocessor, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    return search