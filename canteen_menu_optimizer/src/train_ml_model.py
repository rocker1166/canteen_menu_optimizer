
import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Get base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load preprocessed data
X = pd.read_csv(os.path.join(base_dir, "data/X_preprocessed.csv"))
y = pd.read_csv(os.path.join(base_dir, "data/y_target.csv")).squeeze("columns")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost Regressor
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f"XGBoost Model RMSE: {rmse:.2f}")

# Save the trained model
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, "xgboost_model.pkl"))
print("XGBoost model trained and saved to xgboost_model.pkl")


