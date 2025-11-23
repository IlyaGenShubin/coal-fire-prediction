import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, recall_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def train_and_predict_model(df, feature_cols):
    df = df.dropna(subset=feature_cols + ["цель"])
    X = df[feature_cols].fillna(method="bfill").fillna(0)
    y = df["цель"].astype(int)

    if y.sum() == 0:
        print("Нет пожаров в датасете для обучения.")
        return None, None, None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y if len(np.unique(y)) > 1 else None, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False
    )

    model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val))

    y_pred = model.predict_proba(X_val_scaled)[:, 1]
    ap = average_precision_score(y_val, y_pred)
    rec = recall_score(y_val, model.predict(X_val_scaled))

    print(f"\nCatBoost Model:")
    print(f"Average Precision: {ap:.4f}")
    print(f"Recall: {rec:.4f}")

    return model, scaler, feature_cols
