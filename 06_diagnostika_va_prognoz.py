"""06-amaliy mashg‘ulot. Tibbiy diagnostika va prognoz.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def clinical_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "false_negative": int(fn),
        "false_positive": int(fp),
    }


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y_benign = (data.data, data.target)
    y = 1 - y_benign
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        predictions[name] = prediction
        rows.append({"model": name, **clinical_metrics(y_val, prediction, probability)})
    comparison = pd.DataFrame(rows).set_index("model").round(3)
    print("=== MODEL SOLISHTIRUVI ===")
    print(comparison)
    comparison.to_csv(OUT_DIR / "06_validation_model_comparison.csv")
    best_name = comparison["roc_auc"].idxmax()
    best_model = models[best_name]
    best_prob = best_model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for threshold in np.arange(0.2, 0.81, 0.1):
        pred = (best_prob >= threshold).astype(int)
        metric = clinical_metrics(y_val, pred, best_prob)
        threshold_rows.append(
            {
                "threshold": round(float(threshold), 2),
                "sensitivity": metric["sensitivity"],
                "specificity": metric["specificity"],
                "false_negative": metric["false_negative"],
                "false_positive": metric["false_positive"],
            }
        )
    threshold_table = pd.DataFrame(threshold_rows).round(3)
    print("\n=== THRESHOLD TAHLILI ===")
    print(threshold_table)
    threshold_table.to_csv(OUT_DIR / "06_validation_threshold_analysis.csv", index=False)
    # Model validationda tanlandi; test uchun threshold oldindan 0.50 deb belgilangan.
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= 0.50).astype(int)
    test_metrics = clinical_metrics(y_test, test_prediction, test_probability)
    pd.DataFrame([{"selected_model": best_name, **test_metrics}]).to_csv(
        OUT_DIR / "06_test_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {"row_id": X_test.index, "actual": y_test, "probability": test_probability}
    ).to_csv(
        OUT_DIR / "06_test_predictions.csv",
        index=False,
    )
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, (name, pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred,
            display_labels=["Xavf yo‘q", "Xavf bor"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(
        "Klinik talqin: screening vazifasida false negative ko‘proq zararli bo‘lishi mumkin; threshold tanlovi klinik jamoa bilan validatsiya qilinadi."
    )


if __name__ == "__main__":
    main()
