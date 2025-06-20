import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class EnhancedDecisionEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load enhanced models
        self.ml_model = joblib.load(os.path.join(base_dir, "models/enhanced_xgboost_model.pkl"))
        self.rl_agent_data = joblib.load(os.path.join(base_dir, "models/enhanced_rl_q_table.pkl"))
        self.scaler = joblib.load(os.path.join(base_dir, "models/enhanced_scaler.pkl"))
        self.le_item_id = joblib.load(os.path.join(base_dir, "models/enhanced_le_item_id.pkl"))
        
        # Load data for context
        self.historical_sales = pd.read_csv(os.path.join(base_dir, "data/historical_sales.csv"))
        self.weather_data = pd.read_csv(os.path.join(base_dir, "data/weather_data.csv"))
        self.operational_data = pd.read_csv(os.path.join(base_dir, "data/operational_data.csv"))
        self.academic_data = pd.read_csv(os.path.join(base_dir, "data/academic_calendar.csv"))
        
        # Convert dates
        self.historical_sales["date"] = pd.to_datetime(self.historical_sales["date"])
        self.weather_data["date"] = pd.to_datetime(self.weather_data["date"])
        self.operational_data["date"] = pd.to_datetime(self.operational_data["date"])
        self.academic_data["date"] = pd.to_datetime(self.academic_data["date"])
        
        # Get feature columns from the training data
        self.feature_columns = [
            'day_of_week', 'month', 'day_of_year', 'week_of_year', 'is_weekend',
            'temperature', 'humidity', 'rainfall', 'feels_like_temp', 'student_count',
            'staff_available', 'canteen_capacity', 'event_today', 'hostel_open', 'is_exam_period',
            'is_exam_week', 'is_festival', 'sales_lag_1', 'sales_lag_7', 'sales_3day_avg',
            'sales_same_day_prev_week', 'waste_lag_1', 'item_id_encoded', 'item_popularity_rank',
            'is_monsoon', 'is_winter', 'is_summer', 'temp_humidity_interaction',
            'rain_temp_interaction', 'student_weekend_interaction'
        ]

    def create_enhanced_features(self, date, item_id, current_stock=None, rainfall_today=None, 
                                student_count=None, event_today=None):
        """Create enhanced feature vector for a single prediction"""
        
        pred_date = pd.to_datetime(date)
        
        # Basic temporal features
        features = {
            'day_of_week': pred_date.weekday(),
            'month': pred_date.month,
            'day_of_year': pred_date.dayofyear,
            'week_of_year': pred_date.isocalendar().week,
            'is_weekend': 1 if pred_date.weekday() >= 5 else 0,
        }
        
        # Weather features (use provided or estimate)
        if rainfall_today is not None:
            features['rainfall'] = rainfall_today
            # Estimate other weather based on season and rainfall
            if pred_date.month in [6, 7, 8, 9]:  # Monsoon
                features['temperature'] = 28 + np.random.normal(0, 2)
                features['humidity'] = 85 + np.random.normal(0, 5)
            elif pred_date.month in [12, 1, 2]:  # Winter
                features['temperature'] = 20 + np.random.normal(0, 3)
                features['humidity'] = 65 + np.random.normal(0, 8)
            else:  # Summer/Post-monsoon
                features['temperature'] = 32 + np.random.normal(0, 4)
                features['humidity'] = 60 + np.random.normal(0, 10)
        else:
            # Use historical weather patterns or defaults
            features.update({
                'temperature': 27, 'humidity': 70, 'rainfall': 0
            })
        
        features['feels_like_temp'] = features['temperature'] + (features['humidity'] - 60) * 0.1
        
        # Operational features (use provided or estimate)
        features['student_count'] = student_count if student_count else 250
        features['staff_available'] = 5 if pred_date.weekday() < 5 else 3
        features['canteen_capacity'] = 450 if event_today else 320
        features['event_today'] = event_today if event_today is not None else 0
        features['hostel_open'] = 0 if pred_date.month in [6, 7] else 1
        features['is_exam_period'] = 1 if pred_date.month in [5, 11] else 0
        
        # Academic calendar features
        features['is_exam_week'] = features['is_exam_period']
        features['is_festival'] = 1 if (pred_date.month == 10 and pred_date.day in [12, 13, 14, 15]) else 0
        
        # Historical sales features (get from data)
        item_sales = self.historical_sales[self.historical_sales['item_id'] == item_id].sort_values('date')
        
        if len(item_sales) > 0:
            # Get most recent sales for lagged features
            features['sales_lag_1'] = item_sales.iloc[-1]['quantity_sold'] if len(item_sales) >= 1 else 0
            features['sales_lag_7'] = item_sales.iloc[-7]['quantity_sold'] if len(item_sales) >= 7 else 0
            features['sales_3day_avg'] = item_sales.tail(3)['quantity_sold'].mean() if len(item_sales) >= 3 else 0
            features['sales_same_day_prev_week'] = item_sales.iloc[-7]['quantity_sold'] if len(item_sales) >= 7 else 0
            features['waste_lag_1'] = features['sales_lag_1'] * 0.1  # Estimate 10% waste
        else:
            features.update({
                'sales_lag_1': 0, 'sales_lag_7': 0, 'sales_3day_avg': 0,
                'sales_same_day_prev_week': 0, 'waste_lag_1': 0
            })
        
        # Item features
        try:
            features['item_id_encoded'] = self.le_item_id.transform([item_id])[0]
        except:
            features['item_id_encoded'] = 0
            
        # Item popularity (calculate from historical data)
        item_avg_sales = self.historical_sales.groupby('item_id')['quantity_sold'].mean()
        features['item_popularity_rank'] = item_avg_sales.rank(ascending=False).get(item_id, 5)
        
        # Seasonal features
        features['is_monsoon'] = 1 if pred_date.month in [6, 7, 8, 9] else 0
        features['is_winter'] = 1 if pred_date.month in [12, 1, 2] else 0
        features['is_summer'] = 1 if pred_date.month in [3, 4, 5] else 0
        
        # Interaction features
        features['temp_humidity_interaction'] = features['temperature'] * features['humidity'] / 100
        features['rain_temp_interaction'] = features['rainfall'] * (40 - features['temperature'])
        features['student_weekend_interaction'] = features['student_count'] * features['is_weekend']
        
        return features

    def predict_quantity(self, date, item_id, current_stock=None, rainfall_today=None,
                        student_count=None, event_today=None):
        """Enhanced prediction combining ML and RL with rule-based overrides"""
        
        # Create enhanced features
        features_dict = self.create_enhanced_features(
            date, item_id, current_stock, rainfall_today, student_count, event_today
        )
        
        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])[self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # ML prediction
        ml_prediction = self.ml_model.predict(features_scaled)[0]
        
        # RL adjustment (simplified - using average Q-values)
        rl_adjustment = 0
        if hasattr(self.rl_agent_data, 'get') and 'q_table' in self.rl_agent_data:
            # Get average Q-value as a rough adjustment
            q_values = list(self.rl_agent_data['q_table'].values())
            if q_values:
                avg_q = np.mean([np.max(q) for q in q_values])
                rl_adjustment = avg_q * 0.01  # Small adjustment factor
        
        # Combine predictions
        combined_prediction = ml_prediction + rl_adjustment
        
        # Enhanced rule-based overrides
        final_quantity = combined_prediction
        
        # Rule 1: No stock means no preparation
        if current_stock is not None and current_stock == 0:
            final_quantity = 0
        
        # Rule 2: Heavy rain boosts comfort food (especially maggi, tea)
        if rainfall_today is not None and rainfall_today > 20:
            if 'maggi' in item_id.lower() or 'tea' in item_id.lower():
                final_quantity *= 1.15
        
        # Rule 3: Weekend adjustments
        pred_date = pd.to_datetime(date)
        if pred_date.weekday() >= 5:  # Weekend
            if item_id in ['ice_cream', 'veg_momo']:
                final_quantity *= 1.1  # Popular weekend items
            else:
                final_quantity *= 0.7  # Lower demand overall
        
        # Rule 4: Exam period adjustments
        if features_dict.get('is_exam_period', 0):
            if item_id in ['maggi', 'tea_biscuit']:
                final_quantity *= 1.3  # Study food
            else:
                final_quantity *= 0.9
        
        # Rule 5: Event day adjustments
        if event_today:
            final_quantity *= 1.4
        
        # Rule 6: Seasonal adjustments
        if pred_date.month in [6, 7]:  # Summer vacation
            final_quantity *= 0.4
        
        # Ensure reasonable bounds
        final_quantity = max(0, min(500, final_quantity))
        
        return int(round(final_quantity))

def predict_quantity(date, item_id, current_stock=None, rainfall_today=None,
                    student_count=None, event_today=None):
    """Wrapper function for compatibility"""
    engine = EnhancedDecisionEngine()
    return engine.predict_quantity(date, item_id, current_stock, rainfall_today,
                                 student_count, event_today)

if __name__ == "__main__":
    # Test the enhanced decision engine
    engine = EnhancedDecisionEngine()
    
    test_cases = [
        # (date, item_id, current_stock, rainfall, student_count, event_today)
        ("2024-01-15", "maggi", 50, 25.0, 280, 0),  # Rainy day
        ("2024-01-20", "veg_biryani", 30, 0, 250, 0),  # Normal weekday
        ("2024-01-21", "ice_cream", 20, 0, 150, 0),  # Weekend
        ("2024-05-15", "tea_biscuit", 40, 5.0, 320, 0),  # Exam period
        ("2024-07-01", "fish_curry_rice", 10, 0, 80, 0),  # Summer vacation
    ]
    
    print("Enhanced Decision Engine Test Results:")
    print("=" * 60)
    
    for date, item_id, stock, rain, students, event in test_cases:
        prediction = engine.predict_quantity(date, item_id, stock, rain, students, event)
        print(f"Date: {date}, Item: {item_id}")
        print(f"  Current stock: {stock}, Rainfall: {rain}mm, Students: {students}")
        print(f"  Predicted quantity: {prediction}")
        print()
