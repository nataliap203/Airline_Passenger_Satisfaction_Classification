"""
Eksperyment: TabPFN vs XGBoost vs Las Losowy dla różnych rozmiarów próbek.
Uruchom z poziomu katalogu projektu:
    uv run python run_tabpfn_experiment.py
Wyniki zapisywane do: data/tabpfn_experiment_results.csv
"""

import copy
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, ".")

from src.preprocessing import load_data, clean_data, engineer_features, split_X_y, build_preprocessor
from src.models.tabpfn_model import build_tabpfn_pipeline, train_tabpfn, sample_training_data, predict_proba_batched
from src.models.xgboost_model import build_xgboost_pipeline, train_xgboost
from src.models.random_forest import build_random_forest_pipeline, train_random_forest

_env_sizes = os.environ.get("TABPFN_SAMPLE_SIZES")
SAMPLE_SIZES = [int(s) for s in _env_sizes.split(",")] if _env_sizes else [100, 250, 500]
RANDOM_STATE = 42
XGB_PARAMS = {"max_depth": 7, "learning_rate": 0.1, "n_estimators": 300}
RF_PARAMS = {"max_depth": None, "n_estimators": 500, "min_samples_split": 5}
TABPFN_N_ESTIMATORS = 1
OUTPUT_PATH = "data/tabpfn_experiment_results.csv"


def evaluate_pipeline(pipeline, X_test, y_test, batch_size: int = 256) -> dict:
    y_prob = predict_proba_batched(pipeline, X_test, batch_size=batch_size)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def main():
    print("Ładowanie i przygotowanie danych...")
    train, test = load_data()
    X_train, y_train = split_X_y(engineer_features(clean_data(train)))
    X_test, y_test = split_X_y(engineer_features(clean_data(test)))
    preprocessor = build_preprocessor(X_train)

    n_features_out = preprocessor.fit_transform(X_train).shape[1]
    print(f"X_train: {X_train.shape} → {n_features_out} cech po OHE")
    print(f"X_test:  {X_test.shape}\n")

    records = []

    for sample_size in SAMPLE_SIZES:
        print(f"--- sample_size = {sample_size} ---")
        X_s, y_s = sample_training_data(X_train, y_train, sample_size=sample_size, random_state=RANDOM_STATE)

        tabpfn = build_tabpfn_pipeline(copy.deepcopy(preprocessor), n_estimators=TABPFN_N_ESTIMATORS)
        train_tabpfn(tabpfn, X_s, y_s)
        metrics = evaluate_pipeline(tabpfn, X_test, y_test)
        records.append({"model": "TabPFN", "sample_size": sample_size, **metrics})
        print(f"  TabPFN:  f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

        xgb = build_xgboost_pipeline(copy.deepcopy(preprocessor), **XGB_PARAMS)
        train_xgboost(xgb, X_s, y_s)
        metrics = evaluate_pipeline(xgb, X_test, y_test)
        records.append({"model": "XGBoost", "sample_size": sample_size, **metrics})
        print(f"  XGBoost: f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

        rf = build_random_forest_pipeline(copy.deepcopy(preprocessor), **RF_PARAMS)
        train_random_forest(rf, X_s, y_s)
        metrics = evaluate_pipeline(rf, X_test, y_test)
        records.append({"model": "RF", "sample_size": sample_size, **metrics})
        print(f"  RF:      f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)
    print(f"\nWyniki zapisane do {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
