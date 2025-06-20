from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import date
import logging

from .enhanced_decision_engine import predict_quantity

app = FastAPI(
    title="Enhanced Canteen Menu Optimizer",
    description="AI-powered canteen menu optimization with enhanced features including weather, operational context, and advanced ML/RL models",
    version="2.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictionRequest(BaseModel):
    date: str
    item_id: str
    current_stock: Optional[int] = None
    rainfall_today: Optional[float] = None
    student_count: Optional[int] = None
    event_today: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2024-01-15",
                "item_id": "maggi",
                "current_stock": 50,
                "rainfall_today": 15.5,
                "student_count": 280,
                "event_today": 0
            }
        }

class PredictionResponse(BaseModel):
    item_id: str
    predicted_quantity: int
    model_version: str = "enhanced_v2.0"
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "maggi",
                "predicted_quantity": 125,
                "model_version": "enhanced_v2.0"
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Enhanced Canteen Menu Optimizer API",
        "version": "2.0.0",
        "features": [
            "Enhanced weather context (temperature, humidity, rainfall, feels-like)",
            "Operational context (student count, staff, capacity, events)",
            "Academic calendar integration (holidays, exams, festivals)",
            "Advanced ML model with 30+ features",
            "Reinforcement learning optimization",
            "Intelligent rule-based overrides"
        ],
        "endpoints": {
            "/predict": "POST - Get optimized food quantity prediction",
            "/docs": "GET - API documentation",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "enhanced_v2.0"}

@app.post("/predict", response_model=PredictionResponse)
async def get_enhanced_prediction(request: EnhancedPredictionRequest):
    try:
        logger.info(f"Prediction request for {request.item_id} on {request.date}")
        
        # Validate item_id (basic validation)
        valid_items = [
            "veg_biryani", "fish_curry_rice", "luchi_aloo", "ghugni", "maggi",
            "tea_biscuit", "chicken_roll", "egg_roll", "veg_momo", "ice_cream"
        ]
        
        if request.item_id not in valid_items:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid item_id. Valid items: {', '.join(valid_items)}"
            )
        
        # Get enhanced prediction
        predicted_qty = predict_quantity(
            date=request.date,
            item_id=request.item_id,
            current_stock=request.current_stock,
            rainfall_today=request.rainfall_today,
            student_count=request.student_count,
            event_today=request.event_today
        )
        
        logger.info(f"Predicted {predicted_qty} units for {request.item_id}")
        
        return PredictionResponse(
            item_id=request.item_id,
            predicted_quantity=predicted_qty
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/menu-items")
async def get_menu_items():
    """Get list of available menu items"""
    return {
        "menu_items": [
            {"id": "veg_biryani", "name": "Vegetable Biryani", "price": 50},
            {"id": "fish_curry_rice", "name": "Fish Curry Rice", "price": 80},
            {"id": "luchi_aloo", "name": "Luchi Aloo", "price": 30},
            {"id": "ghugni", "name": "Ghugni", "price": 25},
            {"id": "maggi", "name": "Maggi Noodles", "price": 20},
            {"id": "tea_biscuit", "name": "Tea & Biscuit", "price": 15},
            {"id": "chicken_roll", "name": "Chicken Roll", "price": 40},
            {"id": "egg_roll", "name": "Egg Roll", "price": 35},
            {"id": "veg_momo", "name": "Vegetable Momo", "price": 45},
            {"id": "ice_cream", "name": "Ice Cream", "price": 30}
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the ML models"""
    return {
        "ml_model": {
            "type": "XGBoost Regressor",
            "features": 30,
            "performance": {
                "rmse": 34.39,
                "mae": 26.55,
                "r2_score": 0.6979
            }
        },
        "rl_model": {
            "type": "Enhanced Q-Learning",
            "state_size": 64,
            "action_size": 11,
            "episodes_trained": 150
        },
        "key_features": [
            "is_weekend (58.5% importance)",
            "day_of_week (10.2% importance)",
            "sales_3day_avg (10.2% importance)",
            "student_weekend_interaction (2.5% importance)",
            "sales_lag_1 (2.3% importance)"
        ]
    }
