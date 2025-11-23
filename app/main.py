from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.models import PredictionRequest, PredictionResponse, UploadResponse
from app.data_loader import load_supplies_from_file, load_temperature_from_file, load_fires_from_file
from app.predictor import predict_fire_risk
import os
import pandas as pd

app = FastAPI()

UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.post("/upload-supplies/", response_model=UploadResponse)
async def upload_supplies(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"message": "Файл успешно загружен", "filename": file.filename}

@app.post("/upload-temperature/", response_model=UploadResponse)
async def upload_temperature(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"message": "Файл температуры успешно загружен", "filename": file.filename}

@app.post("/upload-fires/", response_model=UploadResponse)
async def upload_fires(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"message": "Файл пожаров успешно загружен", "filename": file.filename}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Загрузим последние файлы (или используйте постоянные)
    # В реальном приложении — загрузка из базы или кэша
    try:
        temp_path = os.path.join(UPLOAD_DIR, "temperature.csv")
        temp_df = load_temperature_from_file(temp_path)
        supplies_path = os.path.join(UPLOAD_DIR, "supplies.csv")
        sup_df = load_supplies_from_file(supplies_path)
        # Постройте daily, как в обучении
        # В упрощённом варианте: используйте только temp_df
        result = predict_fire_risk(request.sklad, request.shtabel, request.date_str, temp_df, sup_df, pd.DataFrame())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <html>
        <body>
            <h1>Прогнозирование самовозгорания угля</h1>
            <form action="/upload-supplies/" enctype="multipart/form-data" method="post">
                <label>Загрузить supplies.csv</label><br>
                <input name="file" type="file" accept=".csv"><br>
                <input type="submit" value="Загрузить">
            </form>
            <form action="/upload-temperature/" enctype="multipart/form-data" method="post">
                <label>Загрузить temperature.csv</label><br>
                <input name="file" type="file" accept=".csv"><br>
                <input type="submit" value="Загрузить">
            </form>
            <form action="/upload-fires/" enctype="multipart/form-data" method="post">
                <label>Загрузить fires.csv</label><br>
                <input name="file" type="file" accept=".csv"><br>
                <input type="submit" value="Загрузить">
            </form>
            <form action="/predict/" method="post" id="prediction-form">
                <label>Склад:</label><input name="sklad" type="number" required><br>
                <label>Штабель:</label><input name="shtabel" type="number" required><br>
                <label>Дата (YYYY-MM-DD):</label><input name="date_str" type="date" required><br>
                <input type="submit" value="Прогнозировать">
            </form>
            <div id="result"></div>
            <script>
                document.getElementById("prediction-form").onsubmit = async (e) => {
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    const data = Object.fromEntries(formData);
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    document.getElementById("result").innerHTML = `<h3>Риск: ${result.risk} (${(result.probability * 100).toFixed(2)}%)</h3>`;
                };
            </script>
        </body>
    </html>
    """
    return html_content
