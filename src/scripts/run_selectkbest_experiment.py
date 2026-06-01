import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.models.logistic_regression import build_logistic_regression_pipeline

# Najlepsze parametry LR znalezione przez GridSearchCV (l1_ratio: 0.0=L2, 1.0=L1)
_BEST_LR_C = 0.01
_BEST_LR_L1_RATIO = 1.0

_SCORE_FUNC_LABELS: dict = {
    f_classif: "f_classif",
    mutual_info_classif: "mutual_info_classif",
}


def run_selectkbest_experiment(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_values: list | None = None,
    score_funcs: list | None = None,
    random_state: int = 42,
) -> pd.DataFrame:

    if k_values is None:
        k_values = [5, 10, 15, 20, 25, "all"]

    if score_funcs is None:
        score_funcs = [f_classif, mutual_info_classif]

    records = []

    for score_func in score_funcs:
        label = _SCORE_FUNC_LABELS.get(score_func, score_func.__name__)
        print(f"\n── {label} ──")

        for k in k_values:
            pipeline = _build_pipeline(preprocessor, k, score_func, random_state)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            row = {
                "score_func": label,
                "k": k,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
            records.append(row)

            print(
                f"  k={str(k):>4}  accuracy={row['accuracy']:.4f}  "
                f"f1={row['f1']:.4f}  roc_auc={row['roc_auc']:.4f}"
            )

    return pd.DataFrame(records)


def get_feature_scores(
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        score_func: Callable = f_classif,
) -> pd.DataFrame:

    label = _SCORE_FUNC_LABELS.get(score_func, score_func.__name__)
    pipeline = _build_pipeline(preprocessor, k="all", score_func=score_func)
    pipeline.fit(X_train, y_train)

    selector: SelectKBest = pipeline.named_steps["selector"]
    feature_names = _get_feature_names(pipeline)

    p_values = (
        selector.pvalues_
        if hasattr(selector, "pvalues_") and selector.pvalues_ is not None
        else [np.nan] * len(feature_names)
    )

    df = (
        pd.DataFrame({
            "feature": feature_names,
            "score_func": label,
            "score": selector.scores_,
            "p_value": p_values,
        })
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1

    return df


def plot_k_vs_metrics(results: pd.DataFrame) -> None:
    score_funcs = results["score_func"].unique()
    k_labels = [str(k) for k in results[results["score_func"] == score_funcs[0]]["k"]]
    x = np.arange(len(k_labels))

    metrics = ["accuracy", "f1", "roc_auc"]
    colors = {
        "f_classif": "#4575b4",
        "mutual_info_classif": "#d73027",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, m in zip(axes, metrics):
        for sf in score_funcs:
            subset = results[results["score_func"] == sf]
            color = colors.get(sf, "#555")
            ax.plot(x, subset[m].values, "o-", label=sf, color=color, linewidth=2, markersize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(k_labels)
        ax.set_xlabel("k (liczba cech)")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Wpływ liczby wybranych cech i metody selekcji na jakość modelu", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_feature_scores(scores_df: pd.DataFrame, top_n: int = 20) -> None:
    sf = scores_df["score_func"].iloc[0]
    color = "#4575b4"

    subset = scores_df.head(top_n).copy()
    max_score = subset["score"].max()
    subset["score_norm"] = subset["score"] / max_score if max_score > 0 else subset["score"]

    top = subset.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(top["feature"], top["score_norm"], color=color, edgecolor="white", alpha=0.85)

    for bar, rank in zip(bars, top["rank"].values):
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

    x_label = "Wynik F-statystyki (znormalizowany)" if sf == "f_classif" else "Mutual information (znormalizowana)"

    ax.set_xlabel(x_label)
    ax.set_title(f"Top {top_n} najważniejszych cech — {sf}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def _build_pipeline(
    preprocessor: ColumnTransformer,
    k: int | str,
    score_func: Callable = f_classif,
    random_state: int = 42,
) -> Pipeline:
    if score_func is mutual_info_classif:
        score_func = functools.partial(mutual_info_classif, random_state=random_state)

    selector = SelectKBest(score_func=score_func, k=k)
    return build_logistic_regression_pipeline(
        preprocessor,
        C=_BEST_LR_C,
        l1_ratio=_BEST_LR_L1_RATIO,
        max_iter=5000,
        random_state=random_state,
        selector=selector,
    )


def _get_feature_names(pipeline: Pipeline) -> list:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        names = []

        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(cols))
            else:
                names.extend(cols)

        return names
