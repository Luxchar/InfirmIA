from pathlib import Path
from enum import Enum
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "developer_stress_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "developer_stress_scaler.pkl"

# Expected column order after get_dummies(drop_first=True)
FEATURE_COLUMNS = [
    "Hours_Worked",
    "Sleep_Hours",
    "Bugs",
    "Deadline_Days",
    "Coffee_Cups",
    "Meetings",
    "Interruptions",
    "Experience_Years_Mid",
    "Experience_Years_Senior",
    "Code_Complexity_Low",
    "Code_Complexity_Medium",
    "Remote_Work_Yes",
]


class ExperienceLevel(str, Enum):
    junior = "Junior"
    mid = "Mid"
    senior = "Senior"


class CodeComplexity(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class RemoteWork(str, Enum):
    yes = "Yes"
    no = "No"


class StressInput(BaseModel):
    hours_worked: int = Field(..., ge=0, description="Hours worked per day")
    sleep_hours: int = Field(..., ge=0, description="Hours of sleep")
    bugs: int = Field(..., ge=0, description="Number of bugs encountered")
    deadline_days: int = Field(..., ge=0, description="Days until deadline")
    coffee_cups: int = Field(..., ge=0, description="Cups of coffee consumed")
    meetings: int = Field(..., ge=0, description="Number of meetings")
    interruptions: int = Field(..., ge=0, description="Number of interruptions")
    experience_years: ExperienceLevel
    code_complexity: CodeComplexity
    remote_work: RemoteWork


class StressOutput(BaseModel):
    stress_level: float


model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    yield


app = FastAPI(
    title="Developer Stress Predictor",
    description="Predict developer stress level based on work and lifestyle factors.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=StressOutput)
def predict(data: StressInput):
    row = {
        "Hours_Worked": data.hours_worked,
        "Sleep_Hours": data.sleep_hours,
        "Bugs": data.bugs,
        "Deadline_Days": data.deadline_days,
        "Coffee_Cups": data.coffee_cups,
        "Meetings": data.meetings,
        "Interruptions": data.interruptions,
        "Experience_Years": data.experience_years.value,
        "Code_Complexity": data.code_complexity.value,
        "Remote_Work": data.remote_work.value,
    }

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, drop_first=True)
    # Reindex to match training column order, fill missing dummies with 0
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    stress = float(np.clip(prediction, 0, 100))

    return StressOutput(stress_level=round(stress, 2))
