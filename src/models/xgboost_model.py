import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PARAM_GRID = {
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.3],
    "model__n_estimators": [100, 300, 500],
}


def build_xgboost_pipeline(
    preprocessor: ColumnTransformer,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_xgboost(
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
    param_grid: dict = PARAM_GRID,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> GridSearchCV:
    pipeline = build_xgboost_pipeline(preprocessor, random_state=random_state)
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
