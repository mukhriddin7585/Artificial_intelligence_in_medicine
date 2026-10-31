"""11-amaliy mashg‘ulot. Tibbiy startap prototipi.

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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = OUT_DIR


def create_data(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 91, n),
            "length_of_stay": rng.integers(1, 21, n),
            "previous_admissions": rng.poisson(0.9, n),
            "comorbidity_count": rng.poisson(1.8, n),
            "discharge_unit": rng.choice(["Terapiya", "Kardiologiya", "Jarrohlik"], n),
        }
    )
    logit = (
        -3.6
        + 0.025 * df["age"]
        + 0.08 * df["length_of_stay"]
        + 0.48 * df["previous_admissions"]
        + 0.32 * df["comorbidity_count"]
        + rng.normal(0, 0.6, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    df["readmission_30d"] = rng.binomial(1, probability)
    return df


def risk_tier(probability: float) -> str:
    if probability >= 0.6:
        return "Yuqori"
    if probability >= 0.3:
        return "O‘rta"
    return "Past"


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, "Ma’lumot\nEHR"),
        (0.27, "Sifat nazorati\nva belgilar"),
        (0.52, "Model\nvalidatsiyasi"),
        (0.77, "Risk natijasi\nva workflow"),
    ]
    for x, label in boxes:
        ax.text(
            x,
            0.5,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", fc="#9ACFCE", ec="#2F7F89"),
        )
    for x in [0.21, 0.46, 0.71]:
        ax.annotate(
            "",
            xy=(x + 0.04, 0.5),
            xytext=(x, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=2, color="#5367C7"),
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_startup_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = create_data()
    X = df.drop(columns="readmission_30d")
    y = df["readmission_30d"]
    numeric = ["age", "length_of_stay", "previous_admissions", "comorbidity_count"]
    categorical = ["discharge_unit"]
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("prepare", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probability)
    result = X_test.copy()
    result["readmission_probability"] = probability.round(4)
    result["risk_tier"] = result["readmission_probability"].map(risk_tier)
    result.to_csv(OUTPUT_DIR / "11_patient_risk_output.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / "11_readmission_model.joblib")
    kpi = {
        "clinical_problem": "30 kun ichida qayta qabul xavfini erta aniqlash",
        "model_roc_auc": round(float(auc), 3),
        "primary_workflow_kpi": "Yuqori xavfli bemorlarning 24 soatda ko‘rib chiqilish ulushi",
        "safety_kpi": "False negative va subguruh sensitivity monitoringi",
        "human_oversight": "Natijani discharge jamoasi tasdiqlaydi",
    }
    (OUTPUT_DIR / "11_project_kpi.json").write_text(
        json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_pipeline()
    print(json.dumps(kpi, ensure_ascii=False, indent=2))
    print("\nYuqori xavf namunalar:")
    print(result.query("risk_tier == 'Yuqori'").head().to_string(index=False))


if __name__ == "__main__":
    main()
