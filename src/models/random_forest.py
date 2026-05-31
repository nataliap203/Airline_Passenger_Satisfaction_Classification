import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


PARAM_GRID = {
    "model__max_depth": [None, 10, 20],
    "model__n_estimators": [100, 300, 500],
    "model__min_samples_split": [2, 5, 10],
}


def build_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    max_depth: int | None = None,
    n_estimators: int = 300,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:

    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_random_forest(
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

    pipeline = build_random_forest_pipeline(preprocessor, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=4,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    return search