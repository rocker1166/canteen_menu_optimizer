
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load preprocessed data
X = pd.read_csv("canteen_menu_optimizer/data/X_preprocessed.csv")
y = pd.read_csv("canteen_menu_optimizer/data/y_target.csv").squeeze("columns")

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
joblib.dump(model, "canteen_menu_optimizer/models/xgboost_model.pkl")
print("XGBoost model trained and saved to xgboost_model.pkl")


