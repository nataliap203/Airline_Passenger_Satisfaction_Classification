import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= sample_size:
        return X_train, y_train
    X_sampled, _, y_sampled, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=random_state,
    )
    return X_sampled, y_sampled


def build_tabpfn_pipeline(
    preprocessor: ColumnTransformer,
    device: str = "auto",
    n_estimators: int = 4,
) -> Pipeline:
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device=device, n_estimators=n_estimators
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_tabpfn(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_proba_batched(
    pipeline: Pipeline,
    X: pd.DataFrame,
    batch_size: int = 512,
) -> np.ndarray:
    parts = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        parts.append(pipeline.predict_proba(batch))
    return np.vstack(parts)
