"""05-amaliy mashg‘ulot. Mashinali o‘qitish asoslari: sintetik diabetes tasnifi.

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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def create_diabetes_like_data(n: int = 1200) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        weights=[0.66, 0.34],
        class_sep=1.15,
        random_state=RANDOM_STATE,
    )
    names = [
        "glucose",
        "bmi",
        "age",
        "systolic_bp",
        "insulin",
        "pregnancies",
        "family_history",
        "activity_score",
    ]
    df = pd.DataFrame(X, columns=names)
    ranges = {
        "glucose": (110, 30),
        "bmi": (27, 5),
        "age": (48, 15),
        "systolic_bp": (128, 18),
        "insulin": (95, 40),
        "pregnancies": (2.5, 2.2),
        "family_history": (0.5, 0.3),
        "activity_score": (5, 2),
    }
    for column, (mean, std) in ranges.items():
        df[column] = (mean + std * df[column]).round(2)
    df["pregnancies"] = df["pregnancies"].clip(0, 15).round().astype(int)
    df["family_history"] = (df["family_history"] > 0.5).astype(int)
    df["diabetes"] = y
    return df


def main() -> None:
    df = create_diabetes_like_data()
    df.to_csv(OUT_DIR / "05_diabetes_educational_data.csv", index=False)
    X = df.drop(columns="diabetes")
    y = df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print("Train o‘lchami:", X_train.shape)
    print("Test o‘lchami:", X_test.shape)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred, digits=3))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Diabetes yo‘q", "Diabetes xavfi"], cmap="Blues"
    )
    plt.title("Diabetes modeli: confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_diabetes_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    coefficients = pd.Series(pipeline.named_steps["model"].coef_[0], index=X.columns).sort_values()
    coefficients.plot(kind="barh", color="#6D7FD7", figsize=(8, 5))
    plt.title("Standartlashtirilgan belgilar koeffitsiyentlari")
    plt.xlabel("Log-odds koeffitsiyenti")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_diabetes_feature_effects.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("Eslatma: sintetik ma’lumot va model klinik qaror uchun mo‘ljallanmagan.")


if __name__ == "__main__":
    main()
