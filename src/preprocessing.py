import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


def load_data(train_path: str = "train.csv", test_path: str = "test.csv") -> tuple:
    train = pd.read_csv(DATA_DIR / train_path, index_col=0)
    test = pd.read_csv(DATA_DIR / test_path, index_col=0)
    return train, test


def analyze_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Brak brakujących wartości.")
        return
    result = pd.DataFrame(
        {
            "count": missing,
            "percent": (missing / len(df) * 100).round(2),
        }
    )
    print(result.sort_values("percent", ascending=False).to_string())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    df.drop_duplicates(inplace=True)
    return df


def build_preprocessing_pipeline(df: pd.DataFrame) -> Pipeline:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (16, 12)) -> None:
    df_enc = df.copy()
    if "satisfaction" in df_enc.columns:
        df_enc["satisfaction"] = (df_enc["satisfaction"] == "satisfied").astype(int)

    corr = df_enc.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_lower = corr.where(~mask)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_lower, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)

    cols = corr.columns.tolist()
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Macierz korelacji (Pearson) — dolny trójkąt", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_target_correlation(df: pd.DataFrame, target: str = "satisfaction") -> None:
    df_enc = df.copy()
    df_enc[target] = (df_enc[target] == "satisfied").astype(int)

    num_cols = [c for c in df_enc.select_dtypes(include="number").columns if c != target]
    corr_series = df_enc[num_cols].corrwith(df_enc[target]).sort_values()

    colors = ["#d73027" if v < 0 else "#4575b4" for v in corr_series]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(corr_series.index, corr_series.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Korelacja Pearsona z 'satisfaction'")
    ax.set_title("Korelacja cech numerycznych z targetem")
    plt.tight_layout()
    plt.show()


def split_X_y(df: pd.DataFrame, target_col: str = "satisfaction") -> tuple:
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "satisfied").astype(int)
    return X, y


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["avg_service_score"] = df[SERVICE_COLS].mean(axis=1)
    return df
