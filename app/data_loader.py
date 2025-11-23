import pandas as pd
from datetime import timedelta

def load_supplies_from_file(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ["ВыгрузкаНаСклад", "Наим_ЕТСНГ", "Штабель", "ПогрузкаНаСудно", "На_склад_тн", "На_судно_тн", "Склад"]
    for col in ["ВыгрузкаНаСклад", "ПогрузкаНаСудно"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df["На_склад_тн"] = pd.to_numeric(df["На_склад_тн"], errors='coerce')
    df["На_судно_тн"] = pd.to_numeric(df["На_судно_тн"], errors='coerce')
    return df

def load_temperature_from_file(filepath):
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] >= 7:
        df.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта", "Смена"]
    else:
        df.columns = ["Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта"]
    df["Дата акта"] = pd.to_datetime(df["Дата акта"], errors='coerce')
    df["Максимальная температура"] = pd.to_numeric(df["Максимальная температура"], errors='coerce')
    df = df.dropna(subset=["Дата акта", "Максимальная температура"])
    return df

def load_fires_from_file(filepath):
    df = pd.read_csv(filepath)
    df["Дата составления"] = pd.to_datetime(df["Дата составления"], errors='coerce')
    return df
