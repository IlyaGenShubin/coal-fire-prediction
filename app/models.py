from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    sklad: int
    shtabel: int
    date_str: str

class PredictionResponse(BaseModel):
    risk: str
    probability: Optional[float] = None
    message: Optional[str] = None

