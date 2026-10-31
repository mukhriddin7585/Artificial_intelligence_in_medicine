"""02-amaliy mashg‘ulot. Bemor oqimi va telemeditsina.

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

RANDOM_STATE = 42
DATA_FILE = DATA_DIR / "patient_flow.csv"


def create_patient_flow(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    arrival = pd.Timestamp("2026-09-01 08:00") + pd.to_timedelta(
        rng.integers(0, 30 * 24 * 60, n), unit="m"
    )
    channel = rng.choice(["An’anaviy", "Telemeditsina"], n, p=[0.65, 0.35])
    department = rng.choice(
        ["Terapiya", "Kardiologiya", "Nevrologiya", "Pediatriya"], n, p=[0.38, 0.22, 0.17, 0.23]
    )
    priority = rng.choice(["Oddiy", "Shoshilinch"], n, p=[0.85, 0.15])
    base_wait = np.where(channel == "Telemeditsina", 10, 28)
    urgent_adjustment = np.where(priority == "Shoshilinch", -6, 0)
    wait_min = np.maximum(1, base_wait + urgent_adjustment + rng.normal(0, 8, n)).round(0)
    service_min = np.maximum(
        5, np.where(channel == "Telemeditsina", 17, 24) + rng.normal(0, 6, n)
    ).round(0)
    completed = rng.random(n) > np.where(channel == "Telemeditsina", 0.07, 0.04)
    satisfaction = np.clip(5.2 - wait_min / 18 + rng.normal(0, 0.45, n), 1, 5).round(1)
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(1, n + 1)],
            "arrival_time": arrival,
            "channel": channel,
            "department": department,
            "priority": priority,
            "wait_min": wait_min.astype(int),
            "service_min": service_min.astype(int),
            "completed": completed,
            "satisfaction": satisfaction,
        }
    ).sort_values("arrival_time")


def main() -> None:
    if not DATA_FILE.exists():
        create_patient_flow().to_csv(DATA_FILE, index=False)
        print(f"Yangi o‘quv fayli yaratildi: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["arrival_time"])
    df["hour"] = df["arrival_time"].dt.hour
    df["total_cycle_min"] = df["wait_min"] + df["service_min"]
    print("=== MA’LUMOT SIFATI ===")
    print(df.info())
    print("Takror patient_id:", int(df["patient_id"].duplicated().sum()))
    print("Yo‘q qiymatlar:", int(df.isna().sum().sum()))
    channel_summary = (
        df.groupby("channel", observed=True)
        .agg(
            visits=("patient_id", "count"),
            mean_wait_min=("wait_min", "mean"),
            median_cycle_min=("total_cycle_min", "median"),
            completion_rate=("completed", "mean"),
            mean_satisfaction=("satisfaction", "mean"),
        )
        .round(2)
    )
    department_summary = (
        df.groupby("department", observed=True)
        .agg(visits=("patient_id", "count"), mean_wait_min=("wait_min", "mean"))
        .sort_values("mean_wait_min", ascending=False)
        .round(1)
    )
    print("\n=== KANAL BO‘YICHA NATIJA ===")
    print(channel_summary)
    print("\n=== BO‘LIM BO‘YICHA NATIJA ===")
    print(department_summary)
    hourly = df.groupby(["hour", "channel"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    hourly.plot(ax=axes[0], marker="o", color=["#FFB38A", "#4CB8D9"])
    axes[0].set_title("Soatlar bo‘yicha bemor oqimi")
    axes[0].set_xlabel("Soat")
    axes[0].set_ylabel("Murojaatlar soni")
    axes[0].grid(alpha=0.25)
    department_summary["mean_wait_min"].plot(kind="bar", color="#6D7FD7", ax=axes[1])
    axes[1].set_title("Bo‘limlar bo‘yicha o‘rtacha kutish")
    axes[1].set_xlabel("Bo‘lim")
    axes[1].set_ylabel("Daqiqa")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_patient_flow_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    channel_summary.to_csv(OUT_DIR / "02_telemeditsina_case_summary.csv")
    print("\nCase study xulosasi:")
    print(
        "Telemeditsina kanali kutish vaqtini kamaytirishi mumkin, biroq yakunlanmagan seanslar, raqamli tengsizlik va qayta murojaatlar alohida monitoring qilinishi shart."
    )


if __name__ == "__main__":
    main()
