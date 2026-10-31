"""14-amaliy mashg‘ulot. Subguruhlar auditi va psevdonimlashtirish.

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

import hashlib
import hmac
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PSEUDONYM_KEY = os.environ.get("TSI_PSEUDONYM_KEY")
KEY_BYTES = PSEUDONYM_KEY.encode("utf-8") if PSEUDONYM_KEY else os.urandom(32)


def pseudonymize(value: str) -> str:
    return hmac.new(KEY_BYTES, str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def create_data(n: int = 2200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    group = rng.choice(["A", "B"], n, p=[0.68, 0.32])
    age = rng.integers(18, 90, n)
    biomarker = rng.normal(0, 1, n) + np.where(group == "B", -0.2, 0)
    comorbidity = rng.poisson(1.5, n)
    true_logit = -1.4 + 0.025 * (age - 50) + 1.0 * biomarker + 0.25 * comorbidity
    true_prob = 1 / (1 + np.exp(-true_logit))
    outcome = rng.binomial(1, true_prob)
    measured_biomarker = biomarker + np.where(group == "B", rng.normal(-0.45, 0.35, n), 0)
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:05d}" for i in range(n)],
            "group": group,
            "age": age,
            "biomarker": measured_biomarker,
            "comorbidity": comorbidity,
            "outcome": outcome,
        }
    )


def group_metrics(group_df: pd.DataFrame) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        group_df["outcome"], group_df["prediction"], labels=[0, 1]
    ).ravel()
    return {
        "n": len(group_df),
        "selection_rate": group_df["prediction"].mean(),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }


def main() -> None:
    df = create_data()
    train, test = train_test_split(
        df, test_size=0.3, stratify=df[["group", "outcome"]], random_state=RANDOM_STATE
    )
    features = ["age", "biomarker", "comorbidity"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, train["outcome"])
    test = test.copy()
    test["probability"] = model.predict_proba(X_test)[:, 1]
    test["prediction"] = (test["probability"] >= 0.5).astype(int)
    fairness = pd.DataFrame(
        {name: group_metrics(part) for name, part in test.groupby("group")}
    ).T.round(3)
    fairness.index.name = "group"
    rates = fairness.drop(columns="n")
    gap = (rates.max(numeric_only=True) - rates.min(numeric_only=True)).rename("max_min_gap")
    print("=== SUBGURUH METRIKALARI ===")
    print(fairness)
    print("\n=== DISPARITY GAP ===")
    print(gap.round(3))
    fairness.to_csv(OUT_DIR / "14_fairness_metrics.csv")
    gap.to_csv(OUT_DIR / "14_fairness_gaps.csv")
    privacy_export = test[["patient_id", "group", "age", "probability"]].copy()
    privacy_export["patient_key"] = privacy_export["patient_id"].map(pseudonymize)
    privacy_export = privacy_export.drop(columns="patient_id")
    privacy_export.to_csv(OUT_DIR / "14_pseudonymized_results.csv", index=False)
    fairness[["sensitivity", "specificity", "ppv"]].plot(
        kind="bar", figsize=(9, 5), color=["#62C7E8", "#6D7FD7", "#FFB38A"]
    )
    plt.ylim(0, 1)
    plt.ylabel("Ko‘rsatkich")
    plt.title("Subguruhlar bo‘yicha model fairness")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "14_fairness_comparison.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(
        "Privacy tamoyili: minimal zarur maydon, alohida kalit boshqaruvi, kirish nazorati va qayta identifikatsiya xavfi auditi talab qilinadi."
    )


if __name__ == "__main__":
    main()
