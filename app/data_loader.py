import pandas as pd
import numpy as np
from datetime import timedelta

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def load_supplies(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["ВыгрузкаНаСклад", "Наим_ЕТСНГ", "Штабель", "ПогрузкаНаСудно", "На склад, тн", "На судно, тн", "Склад"]
    for col in ["ВыгрузкаНаСклад", "ПогрузкаНаСудно"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df["На склад, тн"] = pd.to_numeric(df["На склад, тн"], errors='coerce')
    df["На судно, тн"] = pd.to_numeric(df["На судно, тн"], errors='coerce')
    return df

def load_fires(path):
    df = pd.read_csv(path)
    df["Дата составления"] = pd.to_datetime(df["Дата составления"], errors='coerce')
    df = df[["Склад", "Штабель", "Дата составления"]].drop_duplicates()
    df = df.rename(columns={"Дата составления": "fire_start"})
    return df

def load_temperature(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] >= 7:
        df.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта", "Смена"]
    else:
        df.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта"]
    df["Дата акта"] = pd.to_datetime(df["Дата акта"], errors='coerce')
    df["Максимальная температура"] = df["Максимальная температура"].apply(safe_float)
    df = df.dropna(subset=["Дата акта", "Максимальная температура"])
    df = df[(df["Максимальная температура"] >= -50) & (df["Максимальная температура"] <= 500)]
    return df

def build_daily_profile(sup, temp, fires, weather_daily, horizon_days=7):
    # Создаём события
    in_events = sup[["Склад", "Штабель", "ВыгрузкаНаСклад", "На склад, тн"]].rename(columns={"ВыгрузкаНаСклад": "date", "На склад, тн": "вес"})
    out_events = sup[["Склад", "Штабель", "ПогрузкаНаСудно", "На судно, тн"]].rename(columns={"ПогрузкаНаСудно": "date", "На судно, тн": "вес"})
    out_events["вес"] = -out_events["вес"]
    events = pd.concat([in_events, out_events], ignore_index=True).dropna(subset=["date", "вес"])
    events["вес_накоп"] = events.groupby(["Склад", "Штабель"])["вес"].cumsum()

    # Ежедневная сетка
    stack_life = events.groupby(["Склад", "Штабель"])["date"].agg(["min", "max"]).reset_index()
    stack_life = stack_life.rename(columns={"min": "start", "max": "end"})

    rows = []
    for _, r in stack_life.iterrows():
        if pd.isna(r["start"]) or pd.isna(r["end"]):
            continue
        dates = pd.date_range(r["start"], r["end"], freq="D")
        for d in dates:
            rows.append({"Склад": r["Склад"], "Штабель": r["Штабель"], "date": d})
    daily = pd.DataFrame(rows)
    daily = daily.merge(events[["Склад", "Штабель", "date", "вес_накоп"]], how="left")
    daily["вес_накоп"] = daily.groupby(["Склад", "Штабель"])["вес_накоп"].ffill().fillna(0)

    # Присоединяем температуру (только до дня date)
    def get_last_temp_before_date(skl, stk, d):
        mask = (temp["Склад"] == skl) & (temp["Штабель"] == stk) & (temp["Дата акта"] < d)
        last = temp[mask].sort_values("Дата акта").tail(1)
        return last["Максимальная температура"].iloc[0] if not last.empty else np.nan

    daily["temp_last"] = daily.apply(lambda row: get_last_temp_before_date(row["Склад"], row["Штабель"], row["date"]), axis=1)

    # Присоединяем погоду
    daily = daily.merge(weather_daily, on="date", how="left")

    # Возраст штабеля
    daily["age_days"] = (daily["date"] - daily.groupby(["Склад", "Штабель"])["date"].transform("min")).dt.days

    # Цель: пожар в течение horizon_days после date
    daily["цель"] = 0
    for _, f in fires.iterrows():
        mask = (
            (daily["Склад"] == f["Склад"]) &
            (daily["Штабель"] == f["Штабель"]) &
            (daily["date"] >= f["fire_start"] - timedelta(days=horizon_days)) &
            (daily["date"] <= f["fire_start"])
        )
        daily.loc[mask, "цель"] = 1

    return daily
