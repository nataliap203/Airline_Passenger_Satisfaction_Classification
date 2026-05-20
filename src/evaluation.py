import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
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
