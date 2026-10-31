"""04-amaliy mashg‘ulot. Tibbiy ma’lumotlarni tozalash.

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

RANDOM_STATE = 42


def create_dataset(n: int = 350) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(n)],
            "age": rng.integers(18, 90, n),
            "bmi": rng.normal(27, 4.5, n).round(1),
            "systolic_bp": rng.normal(128, 17, n).round(0),
            "glucose": rng.normal(105, 25, n).round(1),
            "department": rng.choice(["Terapiya", "Kardiologiya", "Endokrinologiya"], n),
        }
    )
    df.loc[[7, 33], "bmi"] = [4.0, 82.0]
    df.loc[[18, 91], "systolic_bp"] = [20, 360]
    df.loc[[42, 75], "glucose"] = np.nan
    return pd.concat([df, df.iloc[[5, 25]]], ignore_index=True)


def iqr_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ~series.between(q1 - multiplier * iqr, q3 + multiplier * iqr)


def main() -> None:
    raw = create_dataset()
    raw.to_csv(OUT_DIR / "04_raw_medical_data.csv", index=False)
    audit = []
    duplicate_mask = raw.duplicated(subset=["patient_id"], keep="first")
    audit.append(
        ["Duplicate patient_id", int(duplicate_mask.sum()), "Ikkinchi nusxa olib tashlandi"]
    )
    clean = raw.loc[~duplicate_mask].copy()
    for column in ["bmi", "systolic_bp", "glucose"]:
        missing_count = int(clean[column].isna().sum())
        if missing_count:
            median = clean[column].median()
            clean[column] = clean[column].fillna(median)
            audit.append(
                [f"{column}: missing", missing_count, f"Median bilan to‘ldirildi ({median:.1f})"]
            )
    clinical_ranges = {"bmi": (10, 70), "systolic_bp": (60, 260), "glucose": (35, 500)}
    invalid_any = pd.Series(False, index=clean.index)
    for column, (lower, upper) in clinical_ranges.items():
        clinical_invalid = ~clean[column].between(lower, upper)
        statistical_outlier = iqr_mask(clean[column])
        combined = clinical_invalid | statistical_outlier
        audit.append(
            [
                f"{column}: outlier",
                int(combined.sum()),
                "Klinik diapazon va IQR bo‘yicha tekshirildi",
            ]
        )
        invalid_any |= clinical_invalid
    quarantined = clean.loc[invalid_any].copy()
    clean = clean.loc[~invalid_any].copy()
    clean.to_csv(OUT_DIR / "04_clean_medical_data.csv", index=False)
    quarantined.to_csv(OUT_DIR / "04_quarantine_records.csv", index=False)
    pd.DataFrame(audit, columns=["check", "count", "action"]).to_csv(
        OUT_DIR / "04_cleaning_audit.csv", index=False
    )
    print("Xom yozuvlar:", len(raw))
    print("Tozalangan yozuvlar:", len(clean))
    print("Karantinga ajratilgan:", len(quarantined))
    print(pd.DataFrame(audit, columns=["check", "count", "action"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    clean["age"].plot(kind="hist", bins=15, color="#62C7E8", edgecolor="white", ax=axes[0])
    axes[0].set_title("Yosh taqsimoti")
    clean["department"].value_counts().plot(kind="bar", color="#FFB38A", ax=axes[1])
    axes[1].set_title("Bo‘limlar kesimi")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].scatter(clean["bmi"], clean["systolic_bp"], alpha=0.55, color="#5367C7")
    axes[2].set_title("BMI va sistolik bosim")
    axes[2].set_xlabel("BMI")
    axes[2].set_ylabel("Sistolik bosim")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_cleaning_visualizations.png", dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
