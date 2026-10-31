"""12-amaliy mashg‘ulot: modelni o‘qitish, API orqali tekshirish va xizmat.

Standart: python 12_fastapi_tibbiy_model.py
Server:   python 12_fastapi_tibbiy_model.py --serve
Qo‘llanmadagi train_model.py, app.py va test_client.py bitta faylga birlashtirildi.
"""

import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs" / Path(__file__).stem
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"
FEATURES = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]


def train_model():
    """WDBC to‘plamidagi beshta belgidan foydalanuvchi modelni tayyorlaydi."""
    data = load_breast_cancer(as_frame=True)
    X, y = data.data[FEATURES], 1 - data.target  # 1 = malignant, 0 = benign
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    metadata = {
        "model_version": MODEL_VERSION,
        "target": "malignant=1, benign=0",
        "roc_auc_internal_test": float(auc),
        "features": FEATURES,
        "source": "scikit-learn WDBC",
        "intended_use": "Ta’limiy API namoyishi",
        "external_validation": False,
    }
    # Fayl faqat eksport uchun; xizmat shu jarayonda o‘qitilgan modelni ishlatadi.
    joblib.dump(model, OUT_DIR / "12_model.joblib")
    (OUT_DIR / "12_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, metadata


@asynccontextmanager
async def lifespan(application):
    application.state.model, application.state.metadata = train_model()
    yield


app = FastAPI(
    title="Tibbiy risk modeli — ta’limiy API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mean_radius: Annotated[float, Field(gt=0, le=50)]
    mean_texture: Annotated[float, Field(gt=0, le=60)]
    mean_perimeter: Annotated[float, Field(gt=0, le=250)]
    mean_area: Annotated[float, Field(gt=0, le=3000)]
    mean_smoothness: Annotated[float, Field(gt=0, le=1)]


class PredictionResponse(BaseModel):
    model_version: str
    malignant_probability: float
    risk_category: str
    educational_only: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION, "educational_only": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures) -> PredictionResponse:
    supplied = {
        "mean radius": payload.mean_radius,
        "mean texture": payload.mean_texture,
        "mean perimeter": payload.mean_perimeter,
        "mean area": payload.mean_area,
        "mean smoothness": payload.mean_smoothness,
    }
    if set(FEATURES) != set(supplied):
        raise HTTPException(status_code=500, detail="Model sxemasi mos emas")
    row = pd.DataFrame([[supplied[name] for name in FEATURES]], columns=FEATURES)
    probability = float(app.state.model.predict_proba(row)[0, 1])
    category = "Yuqori" if probability >= 0.60 else "O‘rta" if probability >= 0.30 else "Past"
    return PredictionResponse(
        model_version=MODEL_VERSION,
        malignant_probability=round(probability, 4),
        risk_category=category,
        educational_only=True,
    )


def check_api() -> dict:
    """Server portini ochmasdan API so‘rovlari va validatsiyani tekshiradi."""
    payload = {
        "mean_radius": 14.2,
        "mean_texture": 19.0,
        "mean_perimeter": 92.0,
        "mean_area": 655.0,
        "mean_smoothness": 0.10,
    }
    with TestClient(app) as client:
        health_response = client.get("/health")
        response = client.post("/predict", json=payload)
        invalid = client.post("/predict", json={**payload, "mean_radius": -1})
        extra = client.post("/predict", json={**payload, "unknown_field": 1})
        missing = client.post(
            "/predict", json={k: v for k, v in payload.items() if k != "mean_area"}
        )
        assert health_response.status_code == 200
        assert response.status_code == 200
        assert 0 <= response.json()["malignant_probability"] <= 1
        assert invalid.status_code == 422
        assert extra.status_code == 422
        assert missing.status_code == 422
        report = {
            "health": health_response.status_code,
            "valid_request": response.status_code,
            "negative_value": invalid.status_code,
            "unknown_field": extra.status_code,
            "missing_field": missing.status_code,
            "prediction": response.json(),
        }
    (OUT_DIR / "12_api_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve", action="store_true", help="127.0.0.1 manzilida API serverini ochish"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Mahalliy server porti (standart: 8000)"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port 1 dan 65535 gacha bo‘lishi kerak")
    if args.serve:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        print(json.dumps(check_api(), ensure_ascii=False, indent=2))
        print(f"Natijalar: {OUT_DIR}")


if __name__ == "__main__":
    main()
