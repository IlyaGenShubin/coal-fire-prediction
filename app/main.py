from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
from app.data_loader import load_supplies, load_fires, load_temperature, build_daily_profile
from app.predictor import train_and_predict_model
from app.models import PredictionRequest, PredictionResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-and-predict/")
async def upload_and_predict(
    supplies_file: UploadFile = File(...),
    temp_file: UploadFile = File(...),
    fires_file: UploadFile = File(...),
    weather_file: UploadFile = File(...)
):
    # Сохраняем файлы
    supplies_path = os.path.join(UPLOAD_DIR, supplies_file.filename)
    temp_path = os.path.join(UPLOAD_DIR, temp_file.filename)
    fires_path = os.path.join(UPLOAD_DIR, fires_file.filename)
    weather_path = os.path.join(UPLOAD_DIR, weather_file.filename)

    with open(supplies_path, "wb") as f:
        f.write(await supplies_file.read())
    with open(temp_path, "wb") as f:
        f.write(await temp_file.read())
    with open(fires_path, "wb") as f:
        f.write(await fires_file.read())
    with open(weather_path, "wb") as f:
        f.write(await weather_file.read())

    # Загружаем
    sup = load_supplies(supplies_path)
    temp = load_temperature(temp_path)
    fires = load_fires(fires_path)

    # Загружаем погоду
    from app.data_loader import safe_float_weather
    weather = pd.read_csv(weather_path, header=None)
    weather.columns = ["timestamp", "temp_air", "pressure", "humidity", "precip", "wd", "ws", "wg", "cl", "ex1", "ex2"]
    for col in ["temp_air", "pressure", "humidity", "precip"]:
        weather[col] = weather[col].apply(safe_float_weather)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors='coerce')
    weather["date"] = weather["timestamp"].dt.date
    weather["date"] = pd.to_datetime(weather["date"])
    weather_daily = weather.groupby("date").agg({
        "temp_air": "mean",
        "humidity": "mean",
        "precip": "sum",
        "pressure": "mean"
    }).reset_index()

    # Строим daily
    daily = build_daily_profile(sup, temp, fires, weather_daily)

    # Признаки
    feature_cols = [
        "вес_накоп", "age_days", "temp_last",
        "temp_air", "humidity", "precip", "pressure"
    ]
    feature_cols = [c for c in feature_cols if c in daily.columns]

    # Обучаем
    model, scaler, feats = train_and_predict_model(daily, feature_cols)

    if model is None:
        return {"message": "Нет данных для обучения."}

    return {"message": "Модель обучена", "AP": ..., "Recall": ...}
