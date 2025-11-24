import os
from app.services.predictor import train_and_save_model

def ensure_model_trained():
    if not os.path.exists("models/model.pkl"):
        print("Модель не найдена. Обучение...")
        train_and_save_model()
