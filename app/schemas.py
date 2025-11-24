from pydantic import BaseModel
from typing import Dict

class FireRiskRequest(BaseModel):
    sklad: int
    shtabel: int
    date: str  # YYYY-MM-DD

class FireRiskResponse(BaseModel):
    message: str

class MetricsResponse(BaseModel):
    average_precision: float
    recall: float
    feature_importance: Dict[str, float]
