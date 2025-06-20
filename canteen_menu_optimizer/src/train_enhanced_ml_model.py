import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def train_enhanced_ml_model():
    # Load enhanced preprocessed data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X = pd.read_csv(os.path.join(base_dir, "data/X_enhanced_preprocessed.csv"))
    y = pd.read_csv(os.path.join(base_dir, "data/y_enhanced_target.csv")).squeeze("columns")

    print(f"Training with enhanced features: {X.shape[1]} features, {X.shape[0]} samples")
    print("Feature columns:", X.columns.tolist())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Enhanced XGBoost with optimized hyperparameters
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,  # Increased for better learning
        learning_rate=0.08,  # Slightly lower for better generalization
        max_depth=6,  # Increased for complex patterns
        subsample=0.85,  # Slightly higher
        colsample_bytree=0.85,  # Slightly higher
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=42,
        n_jobs=-1  # Use all cores
    )

    print("Training enhanced XGBoost model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Enhanced XGBoost Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    # Feature importance analysis
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    # Save the enhanced model
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "enhanced_xgboost_model.pkl"))
    print("\\nEnhanced XGBoost model trained and saved to enhanced_xgboost_model.pkl")

    # Save feature importance for reference
    importance_df.to_csv(os.path.join(base_dir, "data/feature_importance.csv"), index=False)
    print("Feature importance saved to feature_importance.csv")

    return model, rmse, mae, r2

if __name__ == "__main__":
    model, rmse, mae, r2 = train_enhanced_ml_model()
