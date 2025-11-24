import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, recall_score
from catboost import CatBoostClassifier
import joblib
import os

from app.services.data_loader import load_supplies, load_fires, load_temperature, load_weather

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (для простоты; в проде — кэширование) ===
MODEL = None
SCALER = None
DAILY_DF = None
METRICS = {}
FEATURE_COLS = []

HORIZON_DAYS = 5

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def get_last_temp_before_date(temp_df, skl, stk, d):
    mask = (temp_df["Склад"] == skl) & (temp_df["Штабель"] == stk) & (temp_df["Дата акта"] < d)
    last = temp_df[mask].sort_values("Дата акта").tail(1)
    return last["Максимальная температура"].iloc[0] if not last.empty else np.nan

def prepare_dataset():
    # === этапы из вашего кода ===
    sup = load_supplies()
    fires = load_fires()
    temp = load_temperature()
    weather_daily = load_weather()

    # --- События ---
    in_events = sup[["Склад", "Штабель", "ВыгрузкаНаСклад", "На склад, тн"]].rename(columns={"ВыгрузкаНаСклад": "date", "На склад, тн": "вес"})
    out_events = sup[["Склад", "Штабель", "ПогрузкаНаСудно", "На судно, тн"]].rename(columns={"ПогрузкаНаСудно": "date", "На судно, тн": "вес"})
    out_events["вес"] = -out_events["вес"]
    events = pd.concat([in_events, out_events], ignore_index=True).dropna(subset=["date", "вес"])
    events["вес_накоп"] = events.groupby(["Склад", "Штабель"])["вес"].cumsum()

    # --- Ежедневная сетка ---
    stack_life = events.groupby(["Склад", "Штабель"])["date"].agg(["min", "max"]).reset_index()
    rows = []
    for _, r in stack_life.iterrows():
        if pd.isna(r["start"]) or pd.isna(r["end"]):
            continue
        dates = pd.date_range(r["min"], r["max"], freq="D")
        for d in dates:
            rows.append({"Склад": r["Склад"], "Штабель": r["Штабель"], "date": d})
    daily = pd.DataFrame(rows)
    daily = daily.merge(events[["Склад", "Штабель", "date", "вес_накоп"]], how="left")
    daily["вес_накоп"] = daily.groupby(["Склад", "Штабель"])["вес_накоп"].ffill().fillna(0)

    # --- Добавление штабелей из fires ---
    fires_extra = fires[~fires.set_index(["Склад", "Штабель"]).index.isin(daily.set_index(["Склад", "Штабель"]).index)]
    fires_extra = fires_extra.drop_duplicates(subset=["Склад", "Штабель"])
    for _, f in fires_extra.iterrows():
        start = f["fire_start"] - timedelta(days=14)
        end = f["fire_start"]
        if pd.isna(start) or pd.isna(end):
            continue
        dates = pd.date_range(start, end, freq="D")
        for d in dates:
            daily = pd.concat([daily, pd.DataFrame([{
                "Склад": f["Склад"],
                "Штабель": f["Штабель"],
                "date": d,
                "вес_накоп": np.nan
            }])], ignore_index=True)

    # --- Температура ---
    daily["temp_last"] = daily.apply(lambda row: get_last_temp_before_date(temp, row["Склад"], row["Штабель"], row["date"]), axis=1)

    # --- Погода ---
    daily = daily.merge(weather_daily, on="date", how="left")

    # --- Возраст и цель ---
    daily["age_days"] = (daily["date"] - daily.groupby(["Склад", "Штабель"])["date"].transform("min")).dt.days
    daily["цель"] = 0
    for _, f in fires.iterrows():
        start_window = f["fire_start"] - timedelta(days=HORIZON_DAYS)
        end_window = f["fire_start"]
        mask = (
            (daily["Склад"] == f["Склад"]) &
            (daily["Штабель"] == f["Штабель"]) &
            (daily["date"] >= start_window) &
            (daily["date"] <= end_window)
        )
        daily.loc[mask, "цель"] = 1

    # --- Feature Engineering ---
    daily["temp_trend_7d"] = daily.groupby(["Склад", "Штабель"])["temp_last"].transform(
        lambda x: x.rolling(7, min_periods=3).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=False)
    )
    daily["temp_mean_7d"] = daily.groupby(["Склад", "Штабель"])["temp_last"].transform(lambda x: x.rolling(7, min_periods=3).mean())
    daily["temp_deviation"] = daily["temp_last"] - daily["temp_mean_7d"]
    daily["hot_days_14d"] = daily.groupby(["Склад", "Штабель"])["temp_last"].transform(lambda x: (x > 60).rolling(14, min_periods=1).sum())
    daily["weight_change_7d"] = daily.groupby(["Склад", "Штабель"])["вес_накоп"].transform(lambda x: x.pct_change(7).fillna(0))
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["month"] = daily["date"].dt.month

    global FEATURE_COLS
    FEATURE_COLS = [
        "вес_накоп", "age_days", "temp_last",
        "temp_trend_7d", "temp_deviation", "hot_days_14d",
        "temp_air", "humidity", "pressure", "precip",
        "day_of_year", "month", "weight_change_7d"
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in daily.columns]

    daily = daily.replace([np.inf, -np.inf], np.nan)
    for col in FEATURE_COLS:
        daily[col] = daily.groupby(["Склад", "Штабель"])[col].transform(
            lambda g: g.fillna(method="ffill").fillna(g.mean())
        ).fillna(0)

    daily = daily.dropna(subset=["цель"] + FEATURE_COLS)

    if not weather_daily.empty:
        min_w = weather_daily["date"].min()
        max_w = weather_daily["date"].max()
        daily = daily[(daily["date"] >= min_w) & (daily["date"] <= max_w)]

    return daily

def train_and_save_model():
    global MODEL, SCALER, DAILY_DF, METRICS, FEATURE_COLS
    DAILY_DF = prepare_dataset()
    X = DAILY_DF[FEATURE_COLS].values
    y = DAILY_DF["цель"].values

    if y.sum() == 0:
        raise ValueError("Нет пожаров в обучающих данных")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    SCALER = StandardScaler()
    X_train_scaled = SCALER.fit_transform(X_train)
    X_val_scaled = SCALER.transform(X_val)

    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_pos_weight = neg_count / pos_count

    MODEL = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False
    )
    MODEL.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val))

    y_pred = MODEL.predict_proba(X_val_scaled)[:, 1]
    ap = average_precision_score(y_val, y_pred)
    rec = recall_score(y_val, MODEL.predict(X_val_scaled))

    METRICS = {
        "average_precision": float(ap),
        "recall": float(rec),
        "feature_importance": dict(zip(FEATURE_COLS, MODEL.feature_importances_.tolist()))
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump((MODEL, SCALER, DAILY_DF, METRICS, FEATURE_COLS), "models/model.pkl")

def load_model():
    global MODEL, SCALER, DAILY_DF, METRICS, FEATURE_COLS
    if os.path.exists("models/model.pkl"):
        MODEL, SCALER, DAILY_DF, METRICS, FEATURE_COLS = joblib.load("models/model.pkl")
    else:
        train_and_save_model()

# --- API FUNCTIONS ---
def predict_fire_risk(sklad: str, shtabel: int, date_str: str):
    global MODEL, SCALER, DAILY_DF
    if MODEL is None:
        load_model()
    date = pd.to_datetime(date_str)
    row = DAILY_DF[(DAILY_DF["Склад"] == sklad) & (DAILY_DF["Штабель"] == shtabel) & (DAILY_DF["date"] == date)]
    if row.empty:
        return "Нет данных для указанного штабеля и даты"
    x = row[FEATURE_COLS].values
    x_scaled = SCALER.transform(x)
    prob = MODEL.predict_proba(x_scaled)[0, 1]
    return f"Риск самовозгорания: {prob:.2%}"

def get_fire_calendar():
    global DAILY_DF
    if DAILY_DF is None:
        load_model()
    calendar = DAILY_DF[DAILY_DF["цель"] == 1][["Склад", "Штабель", "date"]].to_dict(orient="records")
    return {"fires": calendar}

def get_model_metrics():
    global METRICS
    if not METRICS:
        load_model()
    return METRICS
