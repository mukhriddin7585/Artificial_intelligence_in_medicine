"""python run_all.py dan keyin ma’lumot va natija shartlarini tekshiradi."""
from pathlib import Path
import json
import itertools
import pandas as pd

ROOT = Path(__file__).resolve().parent
checks = []

def require(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))

folders = sorted(p for p in ROOT.glob("[0-9][0-9]_*") if p.is_dir())
require("15 loyiha mavjud", len(folders) == 15)
for folder in folders:
    code = (folder / "main.py").read_text(encoding="utf-8")
    compile(code, str(folder / "main.py"), "exec")
    nb = load(f"{folder.name}/{folder.name}.ipynb")
    require(folder.name + " notebook kod mosligi",
            "".join(nb["cells"][1]["source"]) == code)
    require(folder.name + " chiqish fayllari mavjud",
            len(list((folder / "outputs").glob("*"))) > 0)
    split_path = folder / "outputs" / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        for (a, x), (b, y) in itertools.combinations(split.items(), 2):
            require(folder.name + f" {a}/{b} ajratilgan", not set(x) & set(y))

data = pd.read_csv(ROOT / "01_Python_Jupyter/data/breast_cancer.csv")
require("WDBC malignant yorlig‘i to‘g‘ri", int(data.malignant.sum()) == 212)
audit = pd.read_csv(ROOT / "03_EHR_sifat_auditi/outputs/row_audit.csv")
require("EHR noto‘g‘ri yoshlar aniqlandi", int(audit.invalid_age.sum()) == 2)
require("EHR noma’lum birlik aniqlandi", int(audit.unknown_unit.sum()) == 1)
clean = load("04_Healthcare_Data_Science/outputs/summary.json")
require("Aynan bir xil 4 nusxa ajratildi", clean["exact_duplicates"] == 4)
require("Diabetes regressiya sifatida berilgan",
        load("05_Diabetes_regressiya/outputs/summary.json")["task"] == "regression")
sensor = pd.read_csv(ROOT / "08_Wearable_monitoring/outputs/monitoring.csv")
require("Yaroqsiz signalda alert yo‘q",
        not sensor.loc[~sensor.evaluable, "demo_alert"].any())
features = pd.read_csv(ROOT / "10_EHR_SQL_Pandas/outputs/features_asof_index.csv")
require("Kelajak sentinel qiymati belgilarga kirmagan",
        not (features.latest_glucose == 999).any())
require("SQL va Pandas bir xil natija berdi",
        load("10_EHR_SQL_Pandas/outputs/summary.json")["sql_matches_pandas"])
api = load("12_FastAPI_xizmati/outputs/api_checks.json")
require("API 200/401/422 shartnomasi", api == {
    "valid": 200, "no_key": 401, "negative_age": 422, "unknown_field": 422})
require("Avtomatik regulator tasnifi berilmagan",
        not load("13_Risk_va_hujjatlar/outputs/summary.json")
        ["regulatory_classification_performed"])
minimal = pd.read_csv(ROOT /
    "14_Fairness_va_maxfiylik/outputs/minimal_pseudonymized.csv")
require("Psevdonim eksportida asl ID yo‘q", "patient_id" not in minimal)
require("HMAC tokenlari yagona", minimal.study_token.nunique() == len(minimal))
require("Tokenlar 256 bitli hex", minimal.study_token.str.len().eq(64).all())
card = load("15_Yakuniy_loyiha/outputs/model_card.json")
require("Threshold validationda tanlangan",
        card["threshold_selected_on"] == "validation")
require("Yakuniy split bemor darajasida", card["split_unit"] == "patient_id")
require("Klinik tashqi validatsiya da’vosi yo‘q", not card["external_validation"])

(ROOT / "contract_checks.json").write_text(json.dumps(
    {"passed": len(checks), "checks": checks}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"{len(checks)} ta mazmuniy tekshiruvdan o‘tdi")
