"""03-amaliy mashg‘ulot. EHR ma’lumotlarining sifat auditi.

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

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = DATA_DIR / "ehr_records.csv"


def create_ehr_file() -> None:
    records = [
        [
            "P001",
            "1984-02-12",
            "2026-09-01",
            "F",
            "120",
            "80",
            "36.7",
            "R51",
            "Bosh og‘rig‘i, 2 kundan beri",
        ],
        [
            "P002",
            "1971-06-03",
            "2026-09-01",
            "M",
            "145",
            "92",
            "37.1",
            "I10",
            "Qon bosimi yuqori, dori qabul qiladi",
        ],
        [
            "P003",
            "2005-11-19",
            "2026-09-02",
            "F",
            "118",
            "76",
            "36.5",
            "J06",
            "Tomoq og‘rig‘i va yo‘tal",
        ],
        [
            "P004",
            "1960-08-28",
            "noto‘g‘ri_sana",
            "M",
            "error",
            "88",
            "38.4",
            "J18",
            "Isitma, hansirash, rentgen tavsiya etildi",
        ],
        ["P005", None, "2026-09-03", "F", "210", "110", None, "I10", "Nazorat ko‘rigi"],
        [
            "P006",
            "1993-04-05",
            "2026-09-03",
            "X",
            "-15",
            "79",
            "36.8",
            None,
            "Erkin klinik izoh mavjud",
        ],
        ["P007", "1948-12-01", "2026-09-04", "M", "132", "85", "45.0", "E11", None],
    ]
    columns = [
        "patient_id",
        "birth_date",
        "visit_date",
        "sex",
        "systolic_bp",
        "diastolic_bp",
        "temperature_c",
        "diagnosis_code",
        "clinical_note",
    ]
    pd.DataFrame(records, columns=columns).to_csv(DATA_FILE, index=False)


def main() -> None:
    if not DATA_FILE.exists():
        create_ehr_file()
    raw = pd.read_csv(DATA_FILE, dtype="string")
    print("=== XOM EHR MA’LUMOTI ===")
    print(raw)
    print("\nXom data type:")
    print(raw.dtypes)
    structured = [
        "patient_id",
        "birth_date",
        "visit_date",
        "sex",
        "systolic_bp",
        "diastolic_bp",
        "temperature_c",
        "diagnosis_code",
    ]
    unstructured = ["clinical_note"]
    print("\nStructured ustunlar:", structured)
    print("Unstructured ustunlar:", unstructured)
    df = raw.copy()
    for column in ["birth_date", "visit_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in ["systolic_bp", "diastolic_bp", "temperature_c"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    missing = df.isna().mean().mul(100).round(1).rename("missing_percent").to_frame()
    validation = pd.DataFrame(
        {
            "rule": [
                "patient_id takrorlanmasligi",
                "visit_date to‘g‘ri sana bo‘lishi",
                "sex M/F qiymatida bo‘lishi",
                "systolic_bp 60–260 oralig‘ida bo‘lishi",
                "temperature_c 30–43 oralig‘ida bo‘lishi",
            ],
            "invalid_rows": [
                int(df["patient_id"].duplicated().sum()),
                int(df["visit_date"].isna().sum()),
                int((~df["sex"].isin(["M", "F"]) & df["sex"].notna()).sum()),
                int((~df["systolic_bp"].between(60, 260) & df["systolic_bp"].notna()).sum()),
                int((~df["temperature_c"].between(30, 43) & df["temperature_c"].notna()).sum()),
            ],
        }
    )
    print("\n=== MISSING VALUE HISOBOTI ===")
    print(missing)
    print("\n=== VALIDATSIYA HISOBOTI ===")
    print(validation)
    issues = []
    for index, row in df.iterrows():
        if pd.isna(row["visit_date"]):
            issues.append([index, row["patient_id"], "visit_date", "Sana parse qilinmadi"])
        if pd.notna(row["systolic_bp"]) and (not 60 <= row["systolic_bp"] <= 260):
            issues.append(
                [index, row["patient_id"], "systolic_bp", "Fiziologik diapazondan tashqari"]
            )
        if pd.notna(row["temperature_c"]) and (not 30 <= row["temperature_c"] <= 43):
            issues.append(
                [index, row["patient_id"], "temperature_c", "Fiziologik diapazondan tashqari"]
            )
    issue_table = pd.DataFrame(issues, columns=["row_index", "patient_id", "field", "problem"])
    issue_table.to_csv(OUT_DIR / "03_ehr_quality_issues.csv", index=False)
    missing.to_csv(OUT_DIR / "03_ehr_missing_report.csv")
    missing.sort_values("missing_percent").plot(
        kind="barh", legend=False, color="#62C7E8", figsize=(8, 5)
    )
    plt.xlabel("Yo‘q qiymatlar, %")
    plt.title("EHR ustunlari bo‘yicha missing value")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_ehr_missing_values.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(
        "\nAudit yakunlandi. Klinik yozuvlar asl nusxasi saqlanib, tuzatishlar alohida logda yuritilishi kerak."
    )


if __name__ == "__main__":
    main()
