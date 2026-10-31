"""13-amaliy mashg‘ulot. Sertifikatlashga tayyorgarlik: risk va dalillar ro‘yxati.

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
import pandas as pd

DEVICE = {
    "name": "Radiologik tasvirda shubhali o‘choqni belgilash tizimi",
    "intended_use_clear": True,
    "makes_autonomous_decision": False,
    "high_severity_condition": True,
    "human_review_required": True,
    "continuous_learning": False,
    "external_validation_complete": False,
    "cybersecurity_plan": True,
    "post_market_monitoring": False,
    "subgroup_evaluation": True,
}
CHECKS = [
    ("Intended use aniq yozilgan", "intended_use_clear", 3, True),
    ("Model mustaqil klinik qaror chiqaradi", "makes_autonomous_decision", 4, False),
    ("Yuqori og‘irlikdagi holatga ta’sir qiladi", "high_severity_condition", 4, False),
    ("Inson nazorati majburiy", "human_review_required", 3, True),
    ("Continuous learning mavjud", "continuous_learning", 3, False),
    ("Tashqi klinik validatsiya yakunlangan", "external_validation_complete", 4, True),
    ("Kiberxavfsizlik rejasi mavjud", "cybersecurity_plan", 3, True),
    ("Post-market monitoring rejasi mavjud", "post_market_monitoring", 4, True),
    ("Subguruhlar bo‘yicha baholangan", "subgroup_evaluation", 3, True),
]


def assess(device: dict) -> pd.DataFrame:
    rows = []
    for criterion, key, weight, desired_value in CHECKS:
        actual = bool(device[key])
        gap = actual != desired_value
        rows.append(
            {
                "criterion": criterion,
                "actual": actual,
                "desired": desired_value,
                "weight": weight,
                "gap": gap,
                "risk_points": weight if gap else 0,
            }
        )
    return pd.DataFrame(rows)


def draw_lifecycle() -> None:
    stages = [
        "Intended use\nva tasnif",
        "Risk boshqaruvi\nva QMS",
        "Analitik/klinik\nvalidatsiya",
        "Regulatory\ntopshirish",
        "Post-market\nmonitoring",
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    for x, stage in zip(xs, stages):
        ax.text(
            x,
            0.5,
            stage,
            transform=ax.transAxes,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.55", fc="#9ACFCE", ec="#2F7F89"),
        )
    for x in [0.17, 0.37, 0.57, 0.77]:
        ax.annotate(
            "",
            xy=(x + 0.04, 0.5),
            xytext=(x, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#5367C7", lw=2),
        )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_regulatory_lifecycle.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    checklist = assess(DEVICE)
    score = int(checklist["risk_points"].sum())
    maximum = int(checklist["weight"].sum())
    ratio = score / maximum
    level = "Yuqori" if ratio >= 0.5 else "O‘rta" if ratio >= 0.2 else "Past"
    print("Qurilma:", DEVICE["name"])
    print(checklist.to_string(index=False))
    print(f"\nIchki o‘quv ustuvorlik balli: {score}/{maximum}; ichki ustuvorlik darajasi: {level}")
    print("Ochiq gaplar:")
    print(checklist.loc[checklist["gap"], ["criterion", "risk_points"]].to_string(index=False))
    checklist.to_csv(OUT_DIR / "13_regulatory_checklist.csv", index=False)
    draw_lifecycle()
    print(
        "Eslatma: bu ta’limiy checklist. Yakuniy klassifikatsiya va sertifikatlash vakolatli regulatory mutaxassis hamda amaldagi hujjatlar bilan aniqlanadi."
    )


if __name__ == "__main__":
    main()
