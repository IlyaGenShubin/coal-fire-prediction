from fastapi import FastAPI, HTTPException
from app.schemas import FireRiskRequest, FireRiskResponse, MetricsResponse
from app.services.predictor import predict_fire_risk, get_fire_calendar, get_model_metrics
from app.utils import ensure_model_trained

app = FastAPI(title="Coal Fire Prediction API")

@app.on_event("startup")
async def startup_event():
    ensure_model_trained()

@app.post("/predict", response_model=FireRiskResponse)
def predict(request: FireRiskRequest):
    try:
        result = predict_fire_risk(request.sklad, request.shtabel, request.date)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/calendar")
def calendar():
    return get_fire_calendar()

@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return get_model_metrics()
