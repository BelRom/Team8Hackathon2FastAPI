import pathlib

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler

from preprocess_pipeline import build_preprocessing_pipeline
from transformers.drop_target import DropTarget

DATA_DIR = pathlib.Path("data")
SESSIONS_FILE = DATA_DIR / "ga_sessions.csv"
HITS_FILE = DATA_DIR / "ga_hits.csv"

MODEL_FILE = pathlib.Path("model_1.joblib")

# всегда просим sklearn‑трансформеры возвращать DataFrame
set_config(transform_output="pandas")

# ─────────────────────────────────────────────────────────────
# Обучение модели
# ─────────────────────────────────────────────────────────────

scaler = MinMaxScaler()
smote = SMOTE(
    k_neighbors=39, sampling_strategy=0.05145,
    random_state=42, n_jobs=-1
)
clf = RandomForestClassifier(
    bootstrap=True, max_depth=18, max_features="log2",
    min_samples_leaf=10, min_samples_split=20,
    n_estimators=600, class_weight="balanced",
    random_state=42, n_jobs=-1
)

def train() -> None:

    df_sessions = pd.read_csv(SESSIONS_FILE, low_memory=False)
    df_hits = pd.read_csv(HITS_FILE, low_memory=False)

    # Базовый препроцессор (fit внутри)
    base_preprocess = build_preprocessing_pipeline(df_hits)
    df_features = base_preprocess.fit_transform(df_sessions)

    # Разделяем X / y
    y = df_features.pop("y").values
    X = df_features

    # Train‑pipeline с SMOTE
    train_pipe = ImbPipeline([
        ("scaler", scaler),
        ("smote", smote),
        ("model", clf)
    ])
    train_pipe.fit(X, y)

    # Берём обученные scaler + model
    trained_scaler = train_pipe.named_steps["scaler"]
    trained_model = train_pipe.named_steps["model"]

    # Продакшен‑пайплайн (без SMOTE)
    serve_pipe = SkPipeline([
        ("preprocess", base_preprocess),  # тот же, что был
        ('drop_target', DropTarget(columns=('y',))),  # ← удаляем 'y'
        ("scaler", trained_scaler),
    ])

    # Сохраняем только то, что нужно на проде
    joblib.dump({
        "preprocessor": serve_pipe,
        "model": trained_model,
        "feature_order": list(X.columns)  # порядок колонок после preprocess
    }, MODEL_FILE)

    print("✅  Прод‑пайплайн сохранён →", MODEL_FILE.resolve())


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: python src/pipeline_fastapi.py train")
