import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def preprocess_enhanced_data(sales_path, weather_path, calendar_path, operational_path):
    # Load all data sources
    sales_df = pd.read_csv(sales_path)
    weather_df = pd.read_csv(weather_path)
    calendar_df = pd.read_csv(calendar_path)
    operational_df = pd.read_csv(operational_path)

    # Convert 'date' columns to datetime objects
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    operational_df["date"] = pd.to_datetime(operational_df["date"])

    # Merge all dataframes
    df = pd.merge(sales_df, weather_df, on="date", how="left")
    df = pd.merge(df, calendar_df, on="date", how="left")
    df = pd.merge(df, operational_df, on="date", how="left")

    # Enhanced Feature Engineering
    # 1. Day Context Features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # 2. Weather Context Features (already included: temperature, humidity, rainfall, feels_like_temp)
    
    # 3. Operational/Logistics Context Features (already included from operational_df)
    
    # 4. Enhanced Sales Context Features
    df = df.sort_values(by=['item_id', 'date'])
    
    # Previous day sales and waste
    df['sales_lag_1'] = df.groupby('item_id')['quantity_sold'].shift(1)
    df['sales_lag_7'] = df.groupby('item_id')['quantity_sold'].shift(7)
    
    # 3-day average sales
    df['sales_3day_avg'] = df.groupby('item_id')['quantity_sold'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # Previous week same day sales (weekly seasonality)
    df['sales_same_day_prev_week'] = df.groupby('item_id')['quantity_sold'].shift(7)
    
    # Calculate waste (assuming some percentage of unsold items become waste)
    df['estimated_waste'] = np.maximum(0, df['quantity_sold'] * np.random.uniform(0.05, 0.15, len(df)))
    df['waste_lag_1'] = df.groupby('item_id')['estimated_waste'].shift(1)
    
    # Revenue and cost features
    df['revenue'] = df['quantity_sold'] * df['price']
    df['total_cost'] = df['quantity_sold'] * df['cost']
    df['profit'] = df['revenue'] - df['total_cost']
    
    # Item popularity rank (based on average sales)
    item_popularity = df.groupby('item_id')['quantity_sold'].mean().rank(ascending=False)
    df['item_popularity_rank'] = df['item_id'].map(item_popularity)
    
    # Seasonal patterns
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)
    
    # Interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['rain_temp_interaction'] = df['rainfall'] * (40 - df['temperature'])  # Cold rain effect
    df['student_weekend_interaction'] = df['student_count'] * df['is_weekend']
    
    # Fill NaN values created by shifting and other operations
    df = df.fillna(0)

    # Encode categorical features
    le_item_id = LabelEncoder()
    df["item_id_encoded"] = le_item_id.fit_transform(df["item_id"])    # Select enhanced features and target
    features = [
        # Day context
        "day_of_week", "month", "day_of_year", "week_of_year", "is_weekend",
        
        # Weather context  
        "temperature", "humidity", "rainfall", "feels_like_temp",
        
        # Operational context (from operational_data.csv)
        "student_count", "staff_available", "canteen_capacity", "event_today",
        "hostel_open", "is_exam_period",
        
        # Academic calendar (from academic_calendar.csv)
        "is_exam_week", "is_festival",
        
        # Note: is_holiday comes from operational_data, not from both sources
        
        # Sales history context
        "sales_lag_1", "sales_lag_7", "sales_3day_avg", "sales_same_day_prev_week",
        "waste_lag_1",
        
        # Item context
        "item_id_encoded", "item_popularity_rank",
        
        # Seasonal patterns
        "is_monsoon", "is_winter", "is_summer",
        
        # Interaction features
        "temp_humidity_interaction", "rain_temp_interaction", "student_weekend_interaction"
    ]
    
    target = "quantity_sold"

    X = df[features]
    y = df[target]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    return X_scaled_df, y, df, scaler, le_item_id

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, 'data/historical_sales.csv')
    weather_path = os.path.join(base_dir, 'data/weather_data.csv')
    calendar_path = os.path.join(base_dir, 'data/academic_calendar.csv')
    operational_path = os.path.join(base_dir, 'data/operational_data.csv')

    X, y, df_full, scaler, le_item_id = preprocess_enhanced_data(sales_path, weather_path, calendar_path, operational_path)
    print("Enhanced data preprocessing complete. Shape of features:", X.shape)
    print("Feature columns:", X.columns.tolist())
    print("First 5 rows of preprocessed features:\n", X.head())
    print("First 5 rows of target:\n", y.head())

    # Save enhanced models and data
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(models_dir, 'enhanced_scaler.pkl'))
    joblib.dump(le_item_id, os.path.join(models_dir, 'enhanced_le_item_id.pkl'))
    print("Enhanced scaler and LabelEncoder saved.")

    # Save enhanced preprocessed data
    data_dir = os.path.join(base_dir, 'data')
    X.to_csv(os.path.join(data_dir, 'X_enhanced_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(data_dir, 'y_enhanced_target.csv'), index=False)
    df_full.to_csv(os.path.join(data_dir, 'full_enhanced_dataset.csv'), index=False)
    print("Enhanced preprocessed features (X) and target (y) saved to CSV.")
