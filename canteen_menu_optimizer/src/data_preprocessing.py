
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def preprocess_data(sales_path, weather_path, calendar_path, operational_path):
    sales_df = pd.read_csv(sales_path)
    weather_df = pd.read_csv(weather_path)
    calendar_df = pd.read_csv(calendar_path)
    operational_df = pd.read_csv(operational_path)

    # Convert 'date' columns to datetime objects
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    operational_df["date"] = pd.to_datetime(operational_df["date"])

    # Merge dataframes
    df = pd.merge(sales_df, weather_df, on="date", how="left")
    df = pd.merge(df, calendar_df, on="date", how="left")
    df = pd.merge(df, operational_df, on="date", how="left")

    # Feature Engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Lagged features for sales (e.g., sales from previous days)
    df = df.sort_values(by=['item_id', 'date'])
    df['sales_lag_1'] = df.groupby('item_id')['quantity_sold'].shift(1)
    df['sales_lag_7'] = df.groupby('item_id')['quantity_sold'].shift(7)
    
    # Add 3-day average sales
    df['sales_3day_avg'] = df.groupby('item_id')['quantity_sold'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # Add waste-related features
    if 'waste_quantity' in df.columns:
        df['waste_lag_1'] = df.groupby('item_id')['waste_quantity'].shift(1)
        df['waste_ratio_lag_1'] = df['waste_lag_1'] / (df['sales_lag_1'] + 1)  # Avoid division by zero
    else:
        df['waste_lag_1'] = 0
        df['waste_ratio_lag_1'] = 0

    # Stock availability features
    stock_columns = [col for col in df.columns if col.startswith('stock_')]
    
    # Create item-specific stock features
    for item in df['item_id'].unique():
        stock_col = f'stock_{item}'
        if stock_col in df.columns:
            df[f'{item}_stock_available'] = df[stock_col]
        else:
            df[f'{item}_stock_available'] = 1  # Default to available

    # Fill NaN values created by shifting and rolling operations
    df = df.fillna(0)

    # Encode categorical features (item_id)
    le_item_id = LabelEncoder()
    df["item_id_encoded"] = le_item_id.fit_transform(df["item_id"])

    # Select enhanced features and target
    base_features = [
        "day_of_week", "month", "day_of_year", "week_of_year",
        "temperature", "humidity", "rainfall", "feels_like_temp",
        "is_holiday", "is_exam_week", "is_festival",
        "is_weekend", "is_exam_period", "is_vacation",
        "student_count", "staff_available", "canteen_capacity",
        "event_today", "hostel_open",
        "sales_lag_1", "sales_lag_7", "sales_3day_avg",
        "waste_lag_1", "waste_ratio_lag_1",
        "item_id_encoded"
    ]
    
    # Add stock features for each item
    item_stock_features = [col for col in df.columns if col.endswith('_stock_available')]
    
    features = base_features + item_stock_features
    
    # Filter features that actually exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    target = "quantity_sold"

    X = df[features]
    y = df[target]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    return X_scaled_df, y, df, scaler, le_item_id

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, 'data/historical_sales.csv')
    weather_path = os.path.join(base_dir, 'data/weather_data.csv')
    calendar_path = os.path.join(base_dir, 'data/academic_calendar.csv')
    operational_path = os.path.join(base_dir, 'data/operational_data.csv')

    X, y, df_full, scaler, le_item_id = preprocess_data(sales_path, weather_path, calendar_path, operational_path)
    print("Enhanced data preprocessing complete. Shape of features:", X.shape)
    print("Feature columns:", X.columns.tolist())
    print("First 5 rows of preprocessed features:\n", X.head())
    print("First 5 rows of target:\n", y.head())

    # Save scaler and label encoder for later use in prediction
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(le_item_id, os.path.join(models_dir, 'le_item_id.pkl'))
    print("Enhanced scaler and LabelEncoder saved.")

    # Save preprocessed data
    data_dir = os.path.join(base_dir, 'data')
    X.to_csv(os.path.join(data_dir, 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(data_dir, 'y_target.csv'), index=False)
    print("Enhanced preprocessed features (X) and target (y) saved to CSV.")


