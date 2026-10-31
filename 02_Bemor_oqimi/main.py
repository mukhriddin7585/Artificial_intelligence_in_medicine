# Bemor oqimi va telemeditsina xizmatini tahlil qilish
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

def main():
    path = DATA / "patient_flow.csv"
    if not path.exists():
        rng = np.random.default_rng(SEED)
        n = 600
        channel = rng.choice(["face_to_face", "telemedicine"], n)
        arrival = pd.Timestamp("2026-01-05 08:00") + pd.to_timedelta(
            rng.integers(0, 5 * 24 * 60, n), unit="m")
        wait = rng.gamma(2, 12, n) + (channel == "face_to_face") * 8
        duration = rng.gamma(3, 5, n)
        start = arrival + pd.to_timedelta(wait, unit="m")
        pd.DataFrame({"visit_id": np.arange(n),
            "department": rng.choice(["A", "B", "C"], n),
            "channel": channel, "arrival_time": arrival,
            "service_start": start,
            "service_end": start + pd.to_timedelta(duration, unit="m")
        }).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["arrival_time",
        "service_start", "service_end"])
    if df.visit_id.duplicated().any():
        raise ValueError("visit_id takrorlangan")
    df["wait_min"] = (df.service_start - df.arrival_time
                      ).dt.total_seconds() / 60
    df["service_min"] = (df.service_end - df.service_start
                         ).dt.total_seconds() / 60
    bad = df[["wait_min", "service_min"]].isna().any(axis=1)
    bad |= (df[["wait_min", "service_min"]] < 0).any(axis=1)
    df.loc[bad].to_csv(OUT / "invalid_rows.csv", index=False)
    clean = df.loc[~bad].copy()
    summary = clean.groupby(["department", "channel"]).agg(
        n=("visit_id", "size"), median_wait=("wait_min", "median"),
        p90_wait=("wait_min", lambda s: s.quantile(0.9)),
        median_service=("service_min", "median"),
        over_30_fraction=("wait_min", lambda s: (s > 30).mean()))
    summary.to_csv(OUT / "department_channel_summary.csv")
    hours = clean.groupby(clean.arrival_time.dt.hour).size()
    hours.to_csv(OUT / "hourly_load.csv", header=["visits"])
    hours.plot.bar(xlabel="Soat", ylabel="Tashriflar soni",
                   title="Bemor oqimi: sun’iy namuna")
    save_plot("hourly_load.png")
    clean.boxplot(column="wait_min", by="channel")
    plt.suptitle("")
    plt.ylabel("Kutish vaqti, daqiqa")
    save_plot("channel_wait.png")
    save_json("summary.json", {"rows": len(df),
        "valid_rows": len(clean), "invalid_rows": int(bad.sum()),
        "busiest_hours": hours.nlargest(3).index.tolist(),
        "data_source": "synthetic"})
    print(summary.to_string())

if __name__ == "__main__":
    main()
