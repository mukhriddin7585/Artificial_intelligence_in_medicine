"""09-amaliy mashg‘ulot. Klinik qarorlarni qo‘llab-quvvatlash tizimi.

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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42


def rule_based_decision(row: pd.Series) -> str:
    if row["spo2"] < 90 or row["systolic_bp"] < 80:
        return "Shoshilinch klinik baholash"
    if row["chest_pain"] == 1 and row["age"] >= 50:
        return "Tezkor EKG va shifokor ko‘rigi"
    if row["temperature_c"] >= 39:
        return "Infeksiya protokoli bo‘yicha baholash"
    return "Standart ko‘rik"


def risk_score(row: pd.Series) -> int:
    score = 0
    score += 2 if row["age"] >= 65 else 0
    score += 3 if row["spo2"] < 92 else 0
    score += 2 if row["systolic_bp"] >= 160 else 0
    score += 2 if row["chest_pain"] == 1 else 0
    score += 1 if row["diabetes"] == 1 else 0
    score += 1 if row["temperature_c"] >= 38 else 0
    return score


def create_patients(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "systolic_bp": rng.normal(132, 22, n).round(0),
            "spo2": rng.normal(96, 2.5, n).clip(82, 100).round(1),
            "temperature_c": rng.normal(36.9, 0.9, n).round(1),
            "chest_pain": rng.binomial(1, 0.18, n),
            "diabetes": rng.binomial(1, 0.22, n),
        }
    )
    latent = (
        0.045 * (df["age"] - 50)
        + 0.09 * (94 - df["spo2"])
        + 0.9 * df["chest_pain"]
        + 0.45 * df["diabetes"]
        + rng.normal(0, 0.9, n)
    )
    df["high_risk"] = (latent > 1.0).astype(int)
    return df


def main() -> None:
    df = create_patients()
    df["rule_recommendation"] = df.apply(rule_based_decision, axis=1)
    df["risk_score"] = df.apply(risk_score, axis=1)
    df["risk_category"] = pd.cut(
        df["risk_score"], bins=[-1, 2, 5, 20], labels=["Past", "O‘rta", "Yuqori"]
    )
    feature_columns = ["age", "systolic_bp", "spo2", "temperature_c", "chest_pain", "diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df["high_risk"],
        test_size=0.25,
        stratify=df["high_risk"],
        random_state=RANDOM_STATE,
    )
    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
    )
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, prediction):.3f}")
    print(classification_report(y_test, prediction, digits=3))
    df.to_csv(OUT_DIR / "09_cdss_patient_results.csv", index=False)
    print("\nQoidaga asoslangan tavsiyalar:")
    print(df["rule_recommendation"].value_counts())
    print("\nRisk kategoriyalari:")
    print(df["risk_category"].value_counts())
    plt.figure(figsize=(15, 7))
    plot_tree(
        tree,
        feature_names=feature_columns,
        class_names=["Past xavf", "Yuqori xavf"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Klinik qarorni qo‘llab-quvvatlovchi Decision Tree")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("CDSS tavsiyasi shifokor qarorini almashtirmaydi; barcha qarorlar loglanishi kerak.")


if __name__ == "__main__":
    main()
