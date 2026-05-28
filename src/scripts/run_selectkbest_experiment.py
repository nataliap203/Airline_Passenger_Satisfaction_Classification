import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def run_selectkbest_experiment(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_values: list[int | str] | None = None,
    score_func: Callable = f_classif,
    random_state: int = 42,
) -> pd.DataFrame:

    if k_values is None:
        k_values = [5, 10, 15, 20, 25, "all"]

    records = []

    for k in k_values:
        pipeline = _build_pipeline(preprocessor, k, score_func, random_state)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        records.append({
            "k": k,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        })

        print(f"k={k} accuracy={records[-1]['accuracy']:.4f} f1={records[-1]['f1']:.4f} roc_auc={records[-1]['roc_auc']:.4f}")

    return pd.DataFrame(records)


def get_feature_scores(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    score_func: Callable = f_classif,
) -> pd.DataFrame:

    pipeline = _build_pipeline(preprocessor, k="all", score_func=score_func)
    pipeline.fit(X_train, y_train)

    selector: SelectKBest = pipeline.named_steps["selector"]
    feature_names = _get_feature_names(pipeline)

    scores_df = pd.DataFrame({
        "feature": feature_names,
        "score": selector.scores_,
        "p_value": selector.pvalues_,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    scores_df["rank"] = scores_df.index + 1

    return scores_df


def plot_k_vs_metrics(results: pd.DataFrame) -> None:
    k_labels = [str(k) for k in results["k"]]
    x = range(len(k_labels))

    fig, ax = plt.subplots(figsize=(9, 5))

    for metric, color, marker in [
        ("accuracy", "#4575b4", "o"),
        ("f1",       "#d73027", "s"),
        ("roc_auc",  "#1a9850", "^"),
    ]:
        ax.plot(x, results[metric], label=metric, color=color,
                marker=marker, linewidth=2, markersize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(k_labels)
    ax.set_xlabel("Liczba wybranych cech (k)")
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Wpływ liczby wybranych cech (SelectKBest) na jakość modelu")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_scores(scores_df: pd.DataFrame, top_n: int = 20) -> None:
    top = scores_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(top["feature"], top["score"], color="#4575b4", edgecolor="white")

    for bar, rank in zip(bars, top["rank"].iloc[::-1]):
        ax.text(bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"#{rank}", va="center", ha="left", fontsize=8, color="white", fontweight="bold")

    ax.set_xlabel("Wynik F-statystyki (f_classif)")
    ax.set_title(f"Top {top_n} najważniejszych cech wg SelectKBest")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def _build_pipeline(
    preprocessor: ColumnTransformer,
    k,
    score_func=f_classif,
    random_state: int = 42,
) -> Pipeline:

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=score_func, k=k)),
        ("model", LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=random_state,
            C=0.01,
            l1_ratio=1.0
        )),
    ])


def _get_feature_names(pipeline: Pipeline) -> list[str]:
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