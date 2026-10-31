# O‘qitilgan model uchun tekshiriladigan FastAPI xizmati
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

from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss, ConfusionMatrixDisplay)

def binary_metrics(y, probability, threshold=0.5):
    pred = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), threshold=float(threshold),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else None,
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability))
            if len(np.unique(y)) == 2 else None,
        brier=float(brier_score_loss(y, probability)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

import os
import secrets
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, ConfigDict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

VERSION = "tsi-12-demo-v1"
API_KEY = os.environ.get("TSI_API_KEY", "local-demo-key-change-me")
FEATURES = ["age", "bmi", "systolic_bp"]

def train_model():
    path = DATA / "api_training.csv"
    if not path.exists():
        rng = np.random.default_rng(SEED)
        n = 500
        frame = pd.DataFrame({"age": rng.integers(18, 90, n),
            "bmi": rng.uniform(18, 40, n),
            "systolic_bp": rng.uniform(90, 180, n)})
        z = ((frame.age - 55) / 25 + (frame.bmi - 27) / 10
             + (frame.systolic_bp - 130) / 40 - 1)
        frame["target"] = rng.binomial(1, 1 / (1 + np.exp(-z)))
        frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    tr, te = train_test_split(df.index, test_size=.2,
        stratify=df.target, random_state=SEED)
    model = make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(df.loc[tr, FEATURES], df.loc[tr, "target"])
    p = model.predict_proba(df.loc[te, FEATURES])[:, 1]
    save_json("metrics.json", binary_metrics(df.loc[te, "target"], p))
    joblib.dump(model, OUT / "model.joblib")
    return model

@asynccontextmanager
async def lifespan(app):
    # Faqat shu loyihada o‘qitilgan model ishlatiladi.
    app.state.model = train_model()
    yield

app = FastAPI(title="TSI o‘quv modeli", version=VERSION, lifespan=lifespan)

class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True,
                              allow_inf_nan=False)
    age: int = Field(ge=0, le=120)
    bmi: float = Field(ge=10, le=80)
    systolic_bp: float = Field(ge=50, le=300)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION,
            "educational_only": True}

@app.post("/predict")
def predict(item: PatientInput, x_api_key: str = Header(default="")):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="API kaliti noto‘g‘ri")
    X = pd.DataFrame([item.model_dump()], columns=FEATURES)
    probability = float(app.state.model.predict_proba(X)[0, 1])
    return {"demo_probability": probability,
        "model_version": VERSION, "educational_only": True}

def main():
    payload = {"age": 60, "bmi": 27.5, "systolic_bp": 135.0}
    headers = {"X-API-Key": API_KEY}
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 200
        assert 0 <= response.json()["demo_probability"] <= 1
        assert client.post("/predict", json=payload).status_code == 401
        invalid = {**payload, "age": -1}
        assert client.post("/predict", json=invalid,
                           headers=headers).status_code == 422
        extra = {**payload, "unknown": 1}
        assert client.post("/predict", json=extra,
                           headers=headers).status_code == 422
        save_json("api_response.json", response.json())
        save_json("api_checks.json", {"valid": 200, "no_key": 401,
            "negative_age": 422, "unknown_field": 422})
        print(response.json())

if __name__ == "__main__":
    main()
