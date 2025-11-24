import pandas as pd
import numpy as np

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def safe_float_weather(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def load_supplies():
    sup = pd.read_csv("data/supplies.csv", header=None)
    sup.columns = ["ВыгрузкаНаСклад", "Наим_ЕТСНГ", "Штабель", "ПогрузкаНаСудно", "На склад, тн", "На судно, тн", "Склад"]
    for col in ["ВыгрузкаНаСклад", "ПогрузкаНаСудно"]:
        sup[col] = pd.to_datetime(sup[col], errors='coerce')
    sup["На склад, тн"] = pd.to_numeric(sup["На склад, тн"], errors='coerce')
    sup["На судно, тн"] = pd.to_numeric(sup["На судно, тн"], errors='coerce')
    return sup

def load_fires():
    # ВАЖНО: fires.csv содержит заголовок!
    fires = pd.read_csv("data/fires.csv")
    fires["Дата составления"] = pd.to_datetime(fires["Дата составления"], errors='coerce')
    return fires[["Склад", "Штабель", "Дата составления"]].rename(columns={"Дата составления": "fire_start"})

def load_temperature():
    temp = pd.read_csv("data/temperature.csv", header=None)
    if temp.shape[1] >= 7:
        temp.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта", "Смена"]
    elif temp.shape[1] == 6:
        temp.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта"]
    else:
        raise ValueError(f"Неподдерживаемое количество столбцов: {temp.shape[1]}")
    temp["Дата акта"] = pd.to_datetime(temp["Дата акта"], errors='coerce')
    temp["Максимальная температура"] = temp["Максимальная температура"].apply(safe_float)
    temp = temp.dropna(subset=["Дата акта", "Максимальная температура"])
    temp = temp[(temp["Максимальная температура"] >= -50) & (temp["Максимальная температура"] <= 500)]
    return temp

def load_weather():
    weather_list = []
    for year in [2019, 2020]:
        try:
            df = pd.read_csv(f"data/weather_data_{year}.csv", header=None)
            if df.shape[1] < 5:
                continue
            df = df.iloc[:, :5]
            df.columns = ["timestamp", "temp_air", "pressure", "humidity", "precip"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.dropna(subset=["timestamp"])
            for col in ["temp_air", "pressure", "humidity", "precip"]:
                df[col] = df[col].apply(safe_float_weather)
            weather_list.append(df)
        except Exception:
            continue
    if weather_list:
        weather = pd.concat(weather_list, ignore_index=True)
        weather["date"] = pd.to_datetime(weather["timestamp"].dt.date)
        return weather.groupby("date").agg({
            "temp_air": "mean",
            "pressure": "mean",
            "humidity": "mean",
            "precip": "sum"
        }).reset_index()
    else:
        return pd.DataFrame(columns=["date", "temp_air", "pressure", "humidity", "precip"])
