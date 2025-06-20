
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import date

from .decision_engine import predict_quantity

app = FastAPI()

class PredictionRequest(BaseModel):
    date: str
    item_id: str
    current_stock: Optional[int] = None
    rainfall_today: Optional[float] = None

@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predicted_qty = predict_quantity(
        request.date,
        request.item_id,
        os.path.join(base_dir, "data/historical_sales.csv"),
        os.path.join(base_dir, "data/weather_data.csv"),
        os.path.join(base_dir, "data/academic_calendar.csv"),
        current_stock=request.current_stock,
        rainfall_today=request.rainfall_today
    )
    return {"item_id": request.item_id, "predicted_quantity": predicted_qty}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


