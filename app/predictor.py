import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# Загрузка модели и скалера (один раз при старте)
model_path = "models/catboost_model.cbm"
model = CatBoostClassifier()
model.load_model(model_path)

# Предположим, что вы сохранили scaler как pickle в models/scaler.pkl
import joblib
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path)

def get_last_temp_before_date(temp_df, skl, stk, d):
    mask = (temp_df["Склад"] == skl) & (temp_df["Штабель"] == stk) & (temp_df["Дата акта"] < d)
    last = temp_df[mask].sort_values("Дата акта").tail(1)
    return last["Максимальная температура"].iloc[0] if not last.empty else np.nan

def predict_fire_risk(sklad, shtabel, date_str, temp_df, supplies_df, weather_df):
    date = pd.to_datetime(date_str)
    
    # Найдём данные для штабеля на этот день
    mask = (supplies_df["Склад"] == sklad) & (supplies_df["Штабель"] == shtabel) & (supplies_df["date"] == date)
    row = supplies_df[mask]
    
    if row.empty:
        return {"risk": "НИЗКИЙ", "probability": 0.0, "message": "Нет данных для прогноза"}

    # Признаки
    weight = row["вес_накоп"].iloc[0]
    age = (date - row.groupby(["Склад", "Штабель"])["date"].transform("min").iloc[0]).days
    temp_last = get_last_temp_before_date(temp_df, sklad, shtabel, date)

    # Добавим остальные признаки (температура, погода и т.д.)
    # Это упрощённый пример; в реальности вычисляются тренды, средние и т.д.
    # Предположим, что у нас есть готовый вектор признаков X
    # В реальном приложении — стройте X так же, как и при обучении модели
    X = np.array([[weight, age, temp_last, 0, 0, 0, 0, 0, 0, 0]])  # 10 признаков
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]
    risk = "ВЫСОКИЙ" if prob > 0.5 else "НИЗКИЙ"
    return {"risk": risk, "probability": prob}
