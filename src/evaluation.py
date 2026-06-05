import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate(model, X_test, y_test) -> pd.Series:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = pd.Series(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
    )

    print(metrics.map("{:.4f}".format).to_string())
    print()
    print(classification_report(y_test, y_pred, target_names=["dissatisfied", "satisfied"]))

    return metrics


def plot_pr_curve(model, X_test: pd.DataFrame, y_test: pd.Series, label: str = "") -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2,
            label=f"{label} (AP={ap:.4f})" if label else f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Krzywa Precision-Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, label: str = "") -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2,
            label=f"{label} (AUC={auc:.4f})" if label else f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Losowy klasyfikator")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Krzywa ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def cross_validate_model(
    model,
    X,
    y,
    cv: int = 5,
    scoring: tuple = ("accuracy", "f1", "roc_auc"),
) -> pd.DataFrame:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    return pd.DataFrame({metric: results[f"test_{metric}"] for metric in scoring})
