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

coal-fire-prediction-main/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI сервер
│   ├── models.py               # DTO (Pydantic)
│   ├── data_loader.py          # Загрузка и объединение данных
│   ├── predictor.py            # Прогноз и обучение модели
│   └── templates/
│       └── index.html          # HTML страница для UI
├── models/
│   └── catboost_model.cbm      # Обученная модель CatBoost (сохранённая)
├── data/                       # (опционально) для хранения CSV
│   ├── supplies.csv            # История поставок/отгрузок угля
│   ├── fires.csv               # Даты самовозгораний
│   ├── temperature.csv         # Температура в штабелях
│   ├── weather_data_2019.csv   # Погода 2019 (часовые данные)
│   ├── weather_data_2020.csv   # Погода 2020 (часовые данные)
│   └── weather_data_2021.csv   # Погода 2021 (часовые данные)
├── requirements.txt            # Зависимости Python
├── Dockerfile                  # Для сборки контейнера
├── docker-compose.yml          # Для запуска через Docker Compose
└── README.md                   # Описание проекта
