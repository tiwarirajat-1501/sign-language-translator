"""
Retrain the SVM with probability=True so confidence gating and adaptive
hold time (config.py) actually work -- the current model wasn't trained
with this flag, so its confidence estimate doesn't reliably separate
correct from wrong predictions (verified: 0.67 vs 0.63 mean confidence
correct/wrong on the current model, vs 0.94 vs 0.50 with probability=True
on the same data).

This mirrors notebooks/05_landmark_normalization.ipynb exactly (same
wrist-centering + scale normalization, same train/test split params,
same SVC hyperparameters) so the result is a drop-in replacement -- just
with usable confidence scores. Nothing about your dataset or model
architecture changes, only this one training flag.

Run from the project root:
    python scripts/retrain_with_probability.py

This will take longer than the original training run -- probability=True
uses an internal 5-fold cross-validation (Platt scaling) to calibrate
the probabilities, which is several times slower than a plain SVC.fit.
For ~74k rows expect several minutes; grab a coffee.
"""

import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "landmarks.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def normalize_row(row):
    """Identical to the normalization in notebooks/05_landmark_normalization.ipynb
    and backend/normalizer.py -- wrist-center, then scale-normalize."""
    landmarks = row.iloc[:63].to_numpy(dtype=np.float64).reshape(21, 3)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    distances = np.linalg.norm(landmarks, axis=1)
    scale = distances.max()
    if scale > 1e-8:
        landmarks = landmarks / scale
    return landmarks.flatten()


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    print("Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("Rows:", len(df), "| Classes:", df["label"].nunique())

    print("Normalizing landmarks (wrist-center + scale)...")
    feature_columns = [c for c in df.columns if c != "label"]
    normalized = np.vstack(df.apply(normalize_row, axis=1).values)
    ndf = pd.DataFrame(normalized, columns=feature_columns)
    ndf["label"] = df["label"].values

    X = ndf[feature_columns]
    y_raw = ndf["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Train samples:", len(X_train), "| Test samples:", len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training SVM with probability=True (this is the slow part)...")
    t0 = time.time()
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    print(f"Training completed in {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=3))

    # confidence separation check -- this is the whole point of retraining
    proba = model.predict_proba(X_test_scaled)
    confidence = proba.max(axis=1)
    correct = y_pred == y_test
    print("Mean confidence when correct:", round(confidence[correct].mean(), 4))
    print("Mean confidence when wrong  :", round(confidence[~correct].mean(), 4))
    print("(bigger gap between these two numbers = more useful for gating)")

    # back up the old model files before overwriting
    backup_dir = MODEL_DIR / "backup_before_probability_retrain"
    backup_dir.mkdir(exist_ok=True)
    for name in ("svm_model.pkl", "scaler.pkl", "label_encoder.pkl"):
        src = MODEL_DIR / name
        if src.exists():
            shutil.copy(src, backup_dir / name)
    print(f"\nOld model files backed up to: {backup_dir}")

    joblib.dump(model, MODEL_DIR / "svm_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print("New model files saved to:", MODEL_DIR)
    print("\nDone. Run backend/realtime_recognition.py -- it should now print")
    print("'Calibrated confidence available' on startup.")


if __name__ == "__main__":
    main()
