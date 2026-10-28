import os
import pandas as pd
from joblib import load
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from schema.payload import GenerateRequest,GenerateResponse

models = {}

@asynccontextmanager
async def lifespan(app : FastAPI):
    models["models"] = load("model_pipeline.pkl")

    yield

    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/predict",response_model=GenerateResponse)
def predict(data : GenerateRequest):
    user_df = pd.DataFrame([data.dict()])

    pred = models["models"].predict(user_df)
    prob = models["models"].predict_proba(user_df)

    return GenerateResponse(
        pred_preciptype = pred,
        prob_preciptype = prob
    )

STATIC_DIR = os.path.join(os.path.devnull(os.path.abspath(__file__)),"static")
app.mount("/",StaticFiles(directory=STATIC_DIR,html=True),name="static")