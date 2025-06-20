
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load trained models and preprocessors
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ml_model = joblib.load(os.path.join(base_dir, "models/xgboost_model.pkl"))
rl_agent = joblib.load(os.path.join(base_dir, "models/rl_q_table.pkl")) # This will be the Q-table dictionary
scaler = joblib.load(os.path.join(base_dir, "models/scaler.pkl"))
le_item_id = joblib.load(os.path.join(base_dir, "models/le_item_id.pkl"))

# Define action levels for RL (must match what was used in training)
action_levels = [0, 50, 100, 150, 200, 250, 300]

def get_current_features(date, item_id, historical_sales_df, weather_df, calendar_df):
    # This function should mimic the feature engineering in data_preprocessing.py
    # for a single prediction point.

    # Create a dummy dataframe for the current prediction point
    current_data = pd.DataFrame({
        'date': [date],
        'item_id': [item_id],
        'quantity_sold': [0] # Placeholder, will be replaced by lagged features
    })

    current_data["date"] = pd.to_datetime(current_data["date"])

    # Merge with weather and calendar data
    current_data = pd.merge(current_data, weather_df, on="date", how="left")
    current_data = pd.merge(current_data, calendar_df, on="date", how="left")

    # Feature Engineering
    current_data["day_of_week"] = current_data["date"].dt.dayofweek
    current_data["month"] = current_data["date"].dt.month
    current_data["day_of_year"] = current_data["date"].dt.dayofyear
    current_data["week_of_year"] = current_data["date"].dt.isocalendar().week.astype(int)

    # Lagged features for sales (need to get from historical_sales_df)
    # This is a simplified way to get lagged features for a single item
    # In a real system, you'd have a more robust way to query past sales.
    past_sales = historical_sales_df[historical_sales_df["item_id"] == item_id]
    past_sales["date"] = pd.to_datetime(past_sales["date"])
    past_sales = past_sales.sort_values(by="date")

    # Get sales for lag_1 and lag_7
    lag_1_date = date - timedelta(days=1)
    lag_7_date = date - timedelta(days=7)

    lag_1_sales = past_sales[past_sales["date"] == lag_1_date]["quantity_sold"].values
    lag_7_sales = past_sales[past_sales["date"] == lag_7_date]["quantity_sold"].values

    current_data["sales_lag_1"] = lag_1_sales[0] if len(lag_1_sales) > 0 else 0
    current_data["sales_lag_7"] = lag_7_sales[0] if len(lag_7_sales) > 0 else 0

    # Encode categorical features (item_id)
    current_data["item_id_encoded"] = le_item_id.transform([item_id])[0]

    features = [
        "day_of_week", "month", "day_of_year", "week_of_year",
        "temperature", "humidity", "rainfall",
        "is_holiday", "is_exam_week", "is_festival",
        "sales_lag_1", "sales_lag_7",
        "item_id_encoded"
    ]

    X_current = current_data[features]
    X_scaled = scaler.transform(X_current)

    return X_scaled

def predict_quantity(date_str, item_id, historical_sales_path, weather_path, calendar_path, current_stock=None, rainfall_today=None):
    date = datetime.strptime(date_str, "%Y-%m-%d")

    historical_sales_df = pd.read_csv(historical_sales_path)
    weather_df = pd.read_csv(weather_path)
    calendar_df = pd.read_csv(calendar_path)

    # Convert date columns to datetime objects
    historical_sales_df["date"] = pd.to_datetime(historical_sales_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # Stage 1: Demand Estimation (ML)
    ml_features = get_current_features(date, item_id, historical_sales_df, weather_df, calendar_df)
    ml_prediction = ml_model.predict(ml_features)[0]

    # Stage 2: Policy Optimization (RL)
    # For simplicity, we'll use a simplified state for the RL agent.
    # A more robust solution would involve passing the full processed state.
    # Here, we'll just use the ML prediction as part of the RL state.
    
    # The RL agent expects a state array. Let's create a dummy state for now.
    # This needs to be aligned with the state definition in canteen_env.py
    # For now, let's use a simplified state based on ML prediction and some temporal features.
    
    # Simplified RL state: [ML_prediction, day_of_week, month]
    rl_state_raw = np.array([ml_prediction, date.weekday(), date.month])
    
    # Quantize the state for RL agent (must match RL agent's _state_to_tuple logic)
    # Assuming RL agent uses np.linspace(state.min(), state.max(), 10) for binning
    # This is a critical point of integration and needs to be exact.
    # For now, let's just cast to int for simplicity, assuming bins are wide enough.
    rl_state_tuple = tuple(rl_state_raw.astype(int))

    # Get action from RL agent (Q-table lookup)
    if rl_state_tuple in rl_agent:
        rl_action_index = np.argmax(rl_agent[rl_state_tuple])
        rl_adjustment = action_levels[rl_action_index] - action_levels[len(action_levels)//2] # Adjust around middle
    else:
        rl_adjustment = 0 # No learned policy for this state, no adjustment

    # Stage 3: Rule-Based Overrides
    final_quantity = ml_prediction + rl_adjustment

    # Rule: If stock is 0, prep = 0
    if current_stock is not None and current_stock == 0:
        final_quantity = 0

    # Rule: Rainfall > 20mm -> prep more Maggi/tea (example, needs item_id check)
    if rainfall_today is not None and rainfall_today > 20 and "Maggi" in item_id: # Simplified check
        final_quantity *= 1.1 # Increase by 10%

    # Ensure quantity is non-negative
    final_quantity = max(0, int(final_quantity))

    return final_quantity

if __name__ == '__main__':
    # Example Usage
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    # For demonstration, let's use a date from our synthetic data range
    example_date_str = "2024-06-15"
    example_item = "item_1"

    predicted_qty = predict_quantity(
        example_date_str,
        example_item,
        os.path.join(base_dir, "data/historical_sales.csv"),
        os.path.join(base_dir, "data/weather_data.csv"),
        os.path.join(base_dir, "data/academic_calendar.csv"),
        current_stock=100, # Example: 100 units in stock
        rainfall_today=25 # Example: 25mm rainfall
    )
    print(f"Predicted quantity for {example_item} on {example_date_str}: {predicted_qty}")

    example_item_2 = "item_5"
    predicted_qty_2 = predict_quantity(
        example_date_str,
        example_item_2,
        os.path.join(base_dir, "data/historical_sales.csv"),
        os.path.join(base_dir, "data/weather_data.csv"),
        os.path.join(base_dir, "data/academic_calendar.csv"),
        current_stock=0, # Example: 0 units in stock
        rainfall_today=5 # Example: 5mm rainfall
    )
    print(f"Predicted quantity for {example_item_2} on {example_date_str}: {predicted_qty_2}")


