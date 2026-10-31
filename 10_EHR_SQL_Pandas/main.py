# EHR tarixidan vaqtga mos belgilar ajratish
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (Path(__file__).resolve().parent
        if "__file__" in globals() else Path.cwd())
DATA, OUT = ROOT / "data", ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SEED = 42

def save_json(name, value):
    text = json.dumps(value, indent=2, ensure_ascii=False,
                      allow_nan=False, default=str)
    (OUT / name).write_text(text, encoding="utf-8")

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=140)
    plt.close()

import sqlite3

INDEX_TIME = "2026-01-15 00:00:00"

def main():
    if not (DATA / "patients.csv").exists():
        rng = np.random.default_rng(SEED)
        patients, visits, labs = [], [], []
        for pid in range(60):
            patients.append({"patient_id": pid,
                "birth_date": f"{1945 + pid % 55}-06-15"})
            for day in [2, 10, 20]:
                date = f"2026-01-{day:02d} 09:00:00"
                visits.append({"visit_id": pid * 3 + day,
                    "patient_id": pid, "visit_date": date,
                    "diagnosis": "SYN_A" if pid % 2 else "SYN_B"})
                labs.append({"patient_id": pid, "lab_time": date,
                    "glucose": 999.0 if day == 20
                        else round(float(rng.uniform(4, 10)), 2)})
        for name, rows in [("patients", patients),
                           ("ehr_visits", visits), ("labs", labs)]:
            pd.DataFrame(rows).to_csv(DATA / f"{name}.csv", index=False)
    patients = pd.read_csv(DATA / "patients.csv")
    visits = pd.read_csv(DATA / "ehr_visits.csv")
    labs = pd.read_csv(DATA / "labs.csv")
    if patients.patient_id.duplicated().any():
        raise ValueError("patients jadvalida primary key buzilgan")
    valid_ids = set(patients.patient_id)
    if not set(visits.patient_id).issubset(valid_ids):
        raise ValueError("visit foreign key xatosi")
    if not set(labs.patient_id).issubset(valid_ids):
        raise ValueError("lab foreign key xatosi")
    with sqlite3.connect(OUT / "ehr.sqlite") as conn:
        for name, frame in [("patients", patients),
                            ("visits", visits), ("labs", labs)]:
            frame.to_sql(name, conn, if_exists="replace", index=False)
        query = """
        WITH counts AS (
          SELECT patient_id, COUNT(*) AS previous_visits
          FROM visits WHERE datetime(visit_date) < datetime(:cutoff)
          GROUP BY patient_id
        ), ranked AS (
          SELECT patient_id, glucose,
            ROW_NUMBER() OVER (
              PARTITION BY patient_id ORDER BY lab_time DESC
            ) AS rn
          FROM labs WHERE datetime(lab_time) < datetime(:cutoff)
        )
        SELECT p.patient_id, COALESCE(c.previous_visits, 0)
          AS previous_visits, r.glucose AS latest_glucose
        FROM patients p LEFT JOIN counts c USING(patient_id)
        LEFT JOIN ranked r ON r.patient_id=p.patient_id AND r.rn=1
        ORDER BY p.patient_id
        """
        sql = pd.read_sql_query(query, conn, params={"cutoff": INDEX_TIME})
    cutoff = pd.Timestamp(INDEX_TIME)
    visits["visit_date"] = pd.to_datetime(visits.visit_date)
    labs["lab_time"] = pd.to_datetime(labs.lab_time)
    previous = visits[visits.visit_date < cutoff]
    counts = previous.groupby("patient_id").size().rename("previous_visits")
    eligible_labs = labs[labs.lab_time < cutoff].sort_values("lab_time")
    latest = eligible_labs.groupby("patient_id").tail(1)
    latest = latest.set_index("patient_id").glucose.rename("latest_glucose")
    features = patients[["patient_id"]].join(counts, on="patient_id")
    features["previous_visits"] = features.previous_visits.fillna(0).astype(int)
    features = features.join(latest, on="patient_id").sort_values("patient_id")
    pd.testing.assert_frame_equal(sql, features.reset_index(drop=True),
                                  check_dtype=False)
    sql.to_csv(OUT / "features_asof_index.csv", index=False)
    (OUT / "feature_query.sql").write_text(query, encoding="utf-8")
    save_json("summary.json", {"patients": len(features),
        "index_time": INDEX_TIME, "sql_matches_pandas": True,
        "future_visits_excluded": int((visits.visit_date >= cutoff).sum()),
        "future_labs_excluded": int((labs.lab_time >= cutoff).sum())})
    print(features.head().to_string(index=False))

if __name__ == "__main__":
    main()
