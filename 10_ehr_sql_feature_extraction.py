"""10-amaliy mashg‘ulot. EHR va SQL asosida belgilar ajratish.

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

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
REFERENCE_DATE = "2026-09-01"
DATABASE = DATA_DIR / "ehr_demo.db"


def build_database(connection: sqlite3.Connection, n_patients: int = 120) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    patient_ids = [f"P{i:04d}" for i in range(1, n_patients + 1)]
    patients = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "birth_date": pd.to_datetime("2026-01-01")
            - pd.to_timedelta(rng.integers(18 * 365, 90 * 365, n_patients), unit="D"),
            "sex": rng.choice(["F", "M"], n_patients),
        }
    )
    encounter_rows, lab_rows, diagnosis_rows = ([], [], [])
    encounter_id = 1
    for patient_id in patient_ids:
        for _ in range(int(rng.integers(1, 7))):
            visit_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(
                int(rng.integers(0, 620)), unit="D"
            )
            encounter_rows.append(
                [
                    encounter_id,
                    patient_id,
                    visit_date,
                    rng.choice(["Ambulator", "Statsionar"], p=[0.85, 0.15]),
                ]
            )
            lab_rows.append(
                [encounter_id, "glucose", round(float(rng.normal(105, 24)), 1), "mg/dL"]
            )
            if rng.random() < 0.25:
                diagnosis_rows.append([encounter_id, rng.choice(["E11", "I10", "J06"])])
            encounter_id += 1
    encounters = pd.DataFrame(
        encounter_rows, columns=["encounter_id", "patient_id", "visit_date", "encounter_type"]
    )
    labs = pd.DataFrame(lab_rows, columns=["encounter_id", "test_name", "value", "unit"])
    diagnoses = pd.DataFrame(diagnosis_rows, columns=["encounter_id", "diagnosis_code"])
    patients.to_sql("patients", connection, if_exists="replace", index=False)
    encounters.to_sql("encounters", connection, if_exists="replace", index=False)
    labs.to_sql("labs", connection, if_exists="replace", index=False)
    diagnoses.to_sql("diagnoses", connection, if_exists="replace", index=False)


def main() -> None:
    with sqlite3.connect(DATABASE) as connection:
        build_database(connection)
        patient_history_sql = """
        SELECT e.patient_id, e.visit_date, e.encounter_type,
               l.test_name, l.value, l.unit, d.diagnosis_code
        FROM encounters AS e
        LEFT JOIN labs AS l ON l.encounter_id = e.encounter_id
        LEFT JOIN diagnoses AS d ON d.encounter_id = e.encounter_id
        WHERE e.patient_id = ?
        ORDER BY e.visit_date
        """
        history = pd.read_sql_query(patient_history_sql, connection, params=["P0001"])
        print("=== P0001 BEMOR TARIXI ===")
        print(history.to_string(index=False))
        feature_sql = """
        WITH latest_glucose AS (
            SELECT e.patient_id, l.value,
                   ROW_NUMBER() OVER (
                       PARTITION BY e.patient_id ORDER BY e.visit_date DESC
                   ) AS rn
            FROM encounters e
            JOIN labs l ON l.encounter_id = e.encounter_id
            WHERE l.test_name = 'glucose' AND datetime(e.visit_date) < datetime(:cutoff)
        ),
        visit_stats AS (
            SELECT patient_id, COUNT(*) AS visit_count,
                   MAX(visit_date) AS last_visit_date
            FROM encounters WHERE datetime(visit_date) < datetime(:cutoff) GROUP BY patient_id
        ),
        diagnosis_flags AS (
            SELECT e.patient_id,
                   MAX(CASE WHEN d.diagnosis_code = 'E11' THEN 1 ELSE 0 END) AS has_e11,
                   MAX(CASE WHEN d.diagnosis_code = 'I10' THEN 1 ELSE 0 END) AS has_i10
            FROM encounters e
            LEFT JOIN diagnoses d ON d.encounter_id = e.encounter_id
            WHERE datetime(e.visit_date) < datetime(:cutoff) GROUP BY e.patient_id
        )
        SELECT p.patient_id, p.birth_date, p.sex,
               v.visit_count, v.last_visit_date,
               g.value AS latest_glucose,
               f.has_e11, f.has_i10
        FROM patients p
        LEFT JOIN visit_stats v ON v.patient_id = p.patient_id
        LEFT JOIN latest_glucose g ON g.patient_id = p.patient_id AND g.rn = 1
        LEFT JOIN diagnosis_flags f ON f.patient_id = p.patient_id
        """
        features = pd.read_sql_query(feature_sql, connection, params={"cutoff": REFERENCE_DATE})
    features["birth_date"] = pd.to_datetime(features["birth_date"])
    features["last_visit_date"] = pd.to_datetime(features["last_visit_date"])
    reference_date = pd.Timestamp(REFERENCE_DATE)
    features["age"] = ((reference_date - features["birth_date"]).dt.days / 365.25).astype(int)
    features["days_since_last_visit"] = (reference_date - features["last_visit_date"]).dt.days
    features["visit_count"] = features["visit_count"].fillna(0).astype(int)
    features[["has_e11", "has_i10"]] = features[["has_e11", "has_i10"]].fillna(0).astype(int)
    model_features = features.drop(columns=["birth_date", "last_visit_date"])
    model_features.to_csv(OUT_DIR / "10_ehr_model_features.csv", index=False)
    print("\n=== FEATURE EXTRACTION NATIJASI ===")
    print(model_features.head().to_string(index=False))
    print("Saqlangan qatorlar:", len(model_features))


if __name__ == "__main__":
    main()
