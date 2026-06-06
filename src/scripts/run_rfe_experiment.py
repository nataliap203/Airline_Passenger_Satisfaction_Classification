import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.models.logistic_regression import build_logistic_regression_pipeline

# Najlepsze parametry LR znalezione przez GridSearchCV (l1_ratio: 0.0=L2, 1.0=L1)
_BEST_LR_C = 0.01
_BEST_LR_L1_RATIO = 1.0


_RFE_ESTIMATORS = {
    "LR": LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
}

N_FEATURES_VALUES = [5, 10, 15, 20, 25, "all"]


def run_rfe_experiment(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_features_values: list[int | str] | None = None,
    estimators: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:

    if n_features_values is None:
        n_features_values = N_FEATURES_VALUES
    if estimators is None:
        estimators = _RFE_ESTIMATORS

    records = []
    for estimator_name, estimator in estimators.items():
        for n_features in n_features_values:
            pipeline = _build_rfe_pipeline(
                copy.deepcopy(preprocessor),
                copy.deepcopy(estimator),
                n_features,
                random_state,
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            record = {
                "estimator": estimator_name,
                "n_features": n_features,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
            records.append(record)
            print(
                f"estimator={estimator_name} n_features={n_features} "
                f"accuracy={record['accuracy']:.4f} f1={record['f1']:.4f} roc_auc={record['roc_auc']:.4f}"
            )

    return pd.DataFrame(records)


def get_feature_rankings(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator_name: str = "LR",
    estimators: dict | None = None,
) -> pd.DataFrame:

    if estimators is None:
        estimators = _RFE_ESTIMATORS

    # n_features_to_select=1 gives full ranking for all features (1=most important)
    pipeline = _build_rfe_pipeline(
        copy.deepcopy(preprocessor),
        copy.deepcopy(estimators[estimator_name]),
        n_features_to_select=1,
    )
    pipeline.fit(X_train, y_train)

    rfe: RFE = pipeline.named_steps["selector"]
    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())

    return (
        pd.DataFrame({"feature": feature_names, "ranking": rfe.ranking_})
        .sort_values("ranking")
        .reset_index(drop=True)
    )


def plot_n_features_vs_metrics(results: pd.DataFrame) -> None:
    n_values = list(dict.fromkeys(results["n_features"]))
    x = range(len(n_values))
    x_labels = [str(n) for n in n_values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, ["accuracy", "f1", "roc_auc"]):
        for estimator_name, group in results.groupby("estimator"):
            y = [group[group["n_features"] == n][metric].values[0] for n in n_values]
            ax.plot(x, y, marker="o", linewidth=2, markersize=7, label=estimator_name)
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Liczba wybranych cech")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Wpływ liczby wybranych cech (RFE) na jakość modelu")
    plt.tight_layout()
    plt.show()


def plot_feature_rankings(
    rankings_df: pd.DataFrame,
    estimator_name: str,
    top_n: int = 20,
) -> None:
    top = rankings_df.head(top_n).iloc[::-1]
    n = len(top)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(top["feature"], n + 1 - top["ranking"], color="#4575b4", edgecolor="white")

    for bar, rank in zip(bars, top["ranking"]):
        ax.text(
            bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"#{rank}",
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            fontweight="bold",
        )

    ax.set_xlabel("Ważność cechy (odwrócony ranking RFE)")
    ax.set_title(f"Top {top_n} najważniejszych cech wg RFE ({estimator_name})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_feature_rankings_comparison(rankings_lr: pd.DataFrame, rankings_rf: pd.DataFrame, top_n: int = 20) -> None:
    df_lr = rankings_lr[["feature", "ranking"]].rename(columns={"ranking": "rank_lr"})
    df_rf = rankings_rf[["feature", "ranking"]].rename(columns={"ranking": "rank_rf"})

    merged = df_lr.merge(df_rf, on="feature")
    top_mask = (merged["rank_lr"] <= top_n) | (merged["rank_rf"] <= top_n)
    merged = merged[top_mask]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged["rank_lr"], merged["rank_rf"], color="#4575b4", alpha=0.7, s=60, zorder=3)

    max_rank = merged[["rank_lr", "rank_rf"]].max().max()
    ax.plot([1, max_rank], [1, max_rank], "k--", linewidth=0.8, alpha=0.5, label="pełna zgodność")

    for _, row in merged.iterrows():
        ax.annotate(
            row["feature"],
            (row["rank_lr"], row["rank_rf"]),
            fontsize=6.5,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Ranga RFE — LR")
    ax.set_ylabel("Ranga RFE — RF")
    ax.set_title(f"Zgodność rankingów RFE: LR vs RF (top {top_n} wg każdego estymatora)")
    ax.legend()
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def _build_rfe_pipeline(
    preprocessor: ColumnTransformer,
    estimator,
    n_features_to_select: int | str,
    random_state: int = 42,
) -> Pipeline:
    selector = None if n_features_to_select == "all" else RFE(
        estimator=estimator, n_features_to_select=n_features_to_select
    )
    return build_logistic_regression_pipeline(
        preprocessor,
        C=_BEST_LR_C,
        l1_ratio=_BEST_LR_L1_RATIO,
        max_iter=5000,
        random_state=random_state,
        selector=selector,
    )
