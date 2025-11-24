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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Прогноз самовозгорания угля</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            input, button { padding: 8px; margin: 5px; font-size: 16px; }
            #result { margin-top: 20px; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Прогноз риска самовозгорания</h1>
        <form id="predictForm">
            Склад: <input type="number" id="sklad" value="4" required><br>
            Штабель: <input type="number" id="shtabel" value="6" required><br>
            Дата (ГГГГ-ММ-ДД): <input type="date" id="date" value="2019-03-11" required><br>
            <button type="submit">Предсказать риск</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('predictForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const data = {
                    sklad: parseInt(document.getElementById('sklad').value),
                    shtabel: parseInt(document.getElementById('shtabel').value),
                    date: document.getElementById('date').value
                };
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                document.getElementById('result').innerText = result.message;
            });
        </script>
    </body>
    </html>
    """
