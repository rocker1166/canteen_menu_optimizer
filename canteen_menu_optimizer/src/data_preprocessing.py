
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(sales_path, weather_path, calendar_path):
    sales_df = pd.read_csv(sales_path)
    weather_df = pd.read_csv(weather_path)
    calendar_df = pd.read_csv(calendar_path)

    # Convert 'date' columns to datetime objects
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # Merge dataframes
    df = pd.merge(sales_df, weather_df, on="date", how="left")
    df = pd.merge(df, calendar_df, on="date", how="left")

    # Feature Engineering
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Lagged features for sales (e.g., sales from previous days)
    df = df.sort_values(by=['item_id', 'date'])
    df['sales_lag_1'] = df.groupby('item_id')['quantity_sold'].shift(1)
    df['sales_lag_7'] = df.groupby('item_id')['quantity_sold'].shift(7)

    # Fill NaN values created by shifting (e.g., with 0 or mean)
    df = df.fillna(0) # Simple fill for now, more sophisticated methods can be used

    # Encode categorical features (item_id, item_name)
    le_item_id = LabelEncoder()
    df["item_id_encoded"] = le_item_id.fit_transform(df["item_id"])

    # Select features and target
    features = [
        "day_of_week", "month", "day_of_year", "week_of_year",
        "temperature", "humidity", "rainfall",
        "is_holiday", "is_exam_week", "is_festival",
        "sales_lag_1", "sales_lag_7",
        "item_id_encoded"
    ]
    target = "quantity_sold"

    X = df[features]
    y = df[target]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    return X_scaled_df, y, df, scaler, le_item_id

if __name__ == '__main__':
    sales_path = 'canteen_menu_optimizer/data/historical_sales.csv'
    weather_path = 'canteen_menu_optimizer/data/weather_data.csv'
    calendar_path = 'canteen_menu_optimizer/data/academic_calendar.csv'

    X, y, df_full, scaler, le_item_id = preprocess_data(sales_path, weather_path, calendar_path)
    print("Data preprocessing complete. Shape of features:", X.shape)
    print("First 5 rows of preprocessed features:\n", X.head())
    print("First 5 rows of target:\n", y.head())

    # Save scaler and label encoder for later use in prediction
    import joblib
    joblib.dump(scaler, 'canteen_menu_optimizer/models/scaler.pkl')
    joblib.dump(le_item_id, 'canteen_menu_optimizer/models/le_item_id.pkl')
    print("Scaler and LabelEncoder saved.")




    X.to_csv("canteen_menu_optimizer/data/X_preprocessed.csv", index=False)
    y.to_csv("canteen_menu_optimizer/data/y_target.csv", index=False)
    print("Preprocessed features (X) and target (y) saved to CSV.")


