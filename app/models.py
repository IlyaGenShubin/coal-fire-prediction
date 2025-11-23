from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    sklad: int
    shtabel: int
    date_str: str

class UploadResponse(BaseModel):
    message: str
    filename: str

class PredictionResponse(BaseModel):
    risk: str
    probability: float
    message: Optional[str] = None
