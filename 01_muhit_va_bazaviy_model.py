"""01-amaliy mashg‘ulot. Python muhiti va bazaviy model.

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

import platform
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
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


def show_environment() -> None:
    """Dasturlash muhiti va kutubxonalar versiyasini chiqaradi."""
    versions = {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Matplotlib": matplotlib.__version__,
        "Scikit-learn": sklearn.__version__,
    }
    print("=== MUHIT TEKSHIRUVI ===")
    for name, version in versions.items():
        print(f"{name:16s}: {version}")


def main() -> None:
    show_environment()
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    target_name = "target"
    print("\n=== MA’LUMOT TO‘PLAMI ===")
    print("O‘lcham:", frame.shape)
    print("Yo‘q qiymatlar:", int(frame.isna().sum().sum()))
    print("Sinflar taqsimoti:")
    print(frame[target_name].value_counts().rename(index=dict(enumerate(dataset.target_names))))
    X = frame.drop(columns=target_name)
    y = frame[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    print("\n=== NATIJALAR ===")
    print(f"Accuracy: {accuracy_score(y_test, prediction):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probability):.3f}")
    print(classification_report(y_test, prediction, target_names=dataset.target_names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frame[target_name].value_counts().sort_index().plot(
        kind="bar", color=["#FFB38A", "#62C7E8"], ax=axes[0]
    )
    axes[0].set_title("Sinflar taqsimoti")
    axes[0].set_xlabel("0 — malignant, 1 — benign")
    axes[0].set_ylabel("Kuzatuvlar soni")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction,
        display_labels=dataset.target_names,
        cmap="Blues",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_bazaviy_model_natijasi.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("Eslatma: model faqat ta’limiy namuna bo‘lib, klinik tashxis o‘rnini bosmaydi.")


if __name__ == "__main__":
    main()
