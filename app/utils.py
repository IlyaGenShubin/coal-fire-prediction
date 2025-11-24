import os
from app.services.predictor import train_and_save_model

def ensure_model_trained():
    if not os.path.exists("models/model.pkl"):
        print("Модель не найдена. Запуск обучения...")
        os.makedirs("models", exist_ok=True)
        train_and_save_model()
    else:
        print("Модель уже обучена.")
