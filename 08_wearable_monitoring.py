"""08-amaliy mashg‘ulot. Raqamli salomatlik va sensor monitoringi.

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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def create_wearable_data(days: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    timestamps = pd.date_range("2026-09-01", periods=days * 24 * 12, freq="5min")
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    circadian = 7 * np.sin(2 * np.pi * (hour - 8) / 24)
    steps = rng.poisson(np.where((hour >= 7) & (hour <= 22), 28, 2))
    heart_rate = 70 + circadian + 0.18 * steps + rng.normal(0, 4, len(timestamps))
    spo2 = 97.2 + rng.normal(0, 0.7, len(timestamps))
    anomaly_positions = [120, 121, 350, 500]
    heart_rate[anomaly_positions[:2]] = [145, 152]
    spo2[anomaly_positions[2:]] = [88, 89]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": heart_rate.round(1),
            "spo2": spo2.round(1),
            "steps": steps,
        }
    )


def main() -> None:
    df = create_wearable_data()
    df["hr_baseline"] = df["heart_rate"].rolling(24, center=True, min_periods=6).median()
    df["hr_deviation"] = (df["heart_rate"] - df["hr_baseline"]).abs()
    df["threshold_anomaly"] = (
        (df["heart_rate"] < 45)
        | (df["heart_rate"] > 130)
        | (df["spo2"] < 92)
        | (df["hr_deviation"] > 35)
    )
    features = df[["heart_rate", "spo2", "steps"]].interpolate().bfill().ffill()
    scaled = StandardScaler().fit_transform(features)
    detector = IsolationForest(n_estimators=300, contamination=0.015, random_state=RANDOM_STATE)
    df["ml_anomaly"] = detector.fit_predict(scaled) == -1
    df["combined_anomaly"] = df["threshold_anomaly"] | df["ml_anomaly"]
    anomalies = df.loc[
        df["combined_anomaly"],
        ["timestamp", "heart_rate", "spo2", "steps", "threshold_anomaly", "ml_anomaly"],
    ]
    print("Topilgan anomaliyalar soni:", len(anomalies))
    print(anomalies.head(20).to_string(index=False))
    df.to_csv(OUT_DIR / "08_wearable_timeseries.csv", index=False)
    anomalies.to_csv(OUT_DIR / "08_wearable_anomalies.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(df["timestamp"], df["heart_rate"], color="#2F7F89", lw=1, label="Yurak urishi")
    axes[0].plot(
        df["timestamp"], df["hr_baseline"], color="#FF9E6D", lw=1.5, label="Rolling baseline"
    )
    selected = df["combined_anomaly"]
    axes[0].scatter(
        df.loc[selected, "timestamp"],
        df.loc[selected, "heart_rate"],
        color="#B51F3B",
        s=28,
        label="Anomaliya",
    )
    axes[0].set_ylabel("bpm")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)
    axes[1].plot(df["timestamp"], df["spo2"], color="#5367C7", lw=1)
    axes[1].axhline(92, color="#B51F3B", linestyle="--", label="92% threshold")
    axes[1].set_ylabel("SpO₂, %")
    axes[1].set_xlabel("Vaqt")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_wearable_monitoring.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("Ogohlantirishlar klinik kontekst va sensor sifati bilan birga tekshiriladi.")


if __name__ == "__main__":
    main()
