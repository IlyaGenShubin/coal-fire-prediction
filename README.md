# Coal Fire Prediction Web App

## Описание

Web-приложение для прогнозирования риска самовозгорания угля на основе:
- Истории поставок/отгрузок (supplies.csv)
- Температур в штабелях (temperature.csv)
- Погоды (weather_data_*.csv)
- Истории пожаров (fires.csv)

## Функционал

- Загрузка CSV-файлов
- Прогноз риска самовозгорания
- Визуализация календаря рисков (в будущем)

## Запуск

```bash
docker-compose up --build

##

coal_fire_prediction/
├── app/
│   ├── __init__.py
│   ├── main.py                 # точка входа FastAPI
│   ├── models.py               # Pydantic модели
│   ├── schemas.py              # схемы запросов/ответов
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # загрузка данных
│   │   └── predictor.py        # прогнозирование
│   └── utils.py                # вспомогательные функции
├── data/
│   ├── fires.csv
│   ├── supplies.csv
│   ├── temperature.csv
│   ├── weather_data_2019.csv
│   └── weather_data_2020.csv
├── models/
│   └── model.pkl               # обученная модель CatBoost (сохраняется автоматически)
├── Dockerfile
├── requirements.txt
├── README.md
└── train_model.py              # скрипт для обучения модели (можно запускать отдельно)
