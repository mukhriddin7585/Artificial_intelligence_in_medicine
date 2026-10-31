"""15-amaliy mashg‘ulot. Yakuniy tibbiy sun’iy intellekt loyihasi.

Tibbiyotda sun’iy intellekt o‘quv qo‘llanmasi ilovasi asosida.
Ishga tushirish va ma’lumotlar tavsifi: README.md.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parent
LESSON_ID = Path(__file__).stem
DATA_DIR = PROJECT_DIR / "data" / LESSON_ID
OUT_DIR = PROJECT_DIR / "outputs" / LESSON_ID
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = OUT_DIR


def prepare_raw_data() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = 1 - df["target"]
    df.insert(0, "patient_id", [f"P{i:04d}" for i in range(len(df))])
    rng = np.random.default_rng(RANDOM_STATE)
    for column in ["mean radius", "mean texture", "mean area"]:
        indices = rng.choice(df.index, size=8, replace=False)
        df.loc[indices, column] = np.nan
    df.loc[df.index[3], "mean radius"] = 999.0
    df = pd.concat([df, df.iloc[[10, 20]]], ignore_index=True)
    return df


def clinical_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "false_negative": int(fn),
        "false_positive": int(fp),
    }


def choose_threshold(y_true, probability) -> float:
    candidates = np.arange(0.2, 0.81, 0.02)
    rows = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        metric = clinical_metrics(y_true, pred, probability)
        utility = 2 * metric["recall_sensitivity"] + metric["specificity"]
        rows.append((threshold, utility))
    return float(max(rows, key=lambda item: item[1])[0])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = prepare_raw_data()
    raw.to_csv(OUTPUT_DIR / "15_raw_data.csv", index=False)
    duplicate_count = int(raw["patient_id"].duplicated().sum())
    clean = raw.drop_duplicates(subset="patient_id", keep="first").copy()
    invalid_radius = clean["mean radius"].gt(60)
    clean.loc[invalid_radius, "mean radius"] = np.nan
    clean["radius_area_ratio"] = clean["mean radius"] / clean["mean area"].replace(0, np.nan)
    clean["texture_perimeter_ratio"] = clean["mean texture"] / clean["mean perimeter"].replace(
        0, np.nan
    )
    y = clean["target"]
    patient_ids = clean["patient_id"]
    X = clean.drop(columns=["patient_id", "target"])
    X_train_full, X_test, y_train_full, y_test, id_train_full, id_test = train_test_split(
        X, y, patient_ids, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE
    )
    split_ids = {
        "train": clean.loc[X_train.index, "patient_id"].tolist(),
        "validation": clean.loc[X_val.index, "patient_id"].tolist(),
        "test": id_test.tolist(),
    }
    assert not set(split_ids["train"]) & set(split_ids["validation"])
    assert not set(split_ids["train"]) & set(split_ids["test"])
    assert not set(split_ids["validation"]) & set(split_ids["test"])
    (OUT_DIR / "15_split.json").write_text(
        json.dumps(split_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    validation_rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        validation_rows.append(
            {"model": name, "validation_roc_auc": roc_auc_score(y_val, val_prob)}
        )
    validation_table = pd.DataFrame(validation_rows).sort_values(
        "validation_roc_auc", ascending=False
    )
    best_name = validation_table.iloc[0]["model"]
    best_model = candidates[best_name]
    val_prob = best_model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob)
    # Validatsiyada tanlangan model va threshold o‘zgartirilmaydi.
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = clinical_metrics(y_test, test_pred, test_prob)
    metrics.update(
        {
            "selected_model": best_name,
            "selected_threshold": round(threshold, 2),
            "duplicates_removed": duplicate_count,
            "invalid_radius_set_missing": int(invalid_radius.sum()),
        }
    )
    predictions = pd.DataFrame(
        {
            "patient_id": id_test.to_numpy(),
            "actual": y_test.to_numpy(),
            "probability": test_prob.round(5),
            "prediction": test_pred,
        }
    )
    predictions["error_type"] = np.select(
        [
            (predictions["actual"] == 1) & (predictions["prediction"] == 0),
            (predictions["actual"] == 0) & (predictions["prediction"] == 1),
        ],
        ["False negative", "False positive"],
        default="To‘g‘ri",
    )
    joblib.dump(best_model, OUTPUT_DIR / "15_best_model.joblib")
    predictions.to_csv(OUTPUT_DIR / "15_test_predictions.csv", index=False)
    validation_table.to_csv(OUTPUT_DIR / "15_model_selection.csv", index=False)
    (OUTPUT_DIR / "15_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = f"MODEL CARD — TA’LIMIY LOYIHA\n    Model: {best_name}\n    Maqsad: malignant xavfni ta’limiy tasniflash\n    Test ROC-AUC: {metrics['roc_auc']:.3f}\n    Sensitivity: {metrics['recall_sensitivity']:.3f}\n    Specificity: {metrics['specificity']:.3f}\n    Threshold: {threshold:.2f}\n    Cheklov: ochiq o‘quv dataseti; tashqi va prospektiv klinik validatsiya bajarilmagan.\n    Nazorat: natija shifokor qarorini almashtirmaydi.\n    "
    (OUTPUT_DIR / "15_model_card.txt").write_text(model_card, encoding="utf-8")
    print(validation_table.to_string(index=False))
    print("\nTest metrikalari:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nXatolar taqsimoti:")
    print(predictions["error_type"].value_counts())
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=["Benign", "Malignant"], cmap="Blues"
    )
    plt.title(f"{best_name} — test confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
