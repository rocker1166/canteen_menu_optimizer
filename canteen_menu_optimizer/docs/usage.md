# Canteen Menu Optimizer - Usage Guide

This document provides detailed instructions for using the Canteen Menu Optimizer API.

## Setup and Running the API Server

Before using the API, ensure you have completed the setup as described in the main README file.

### Quick Start

1. Activate the virtual environment:
   ```bash
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. Start the API server:
   ```bash
   python -m uvicorn src.api_backend:app --host 0.0.0.0 --port 8000
   ```

3. The API will be accessible at `http://0.0.0.0:8000` or `http://localhost:8000`

## API Endpoints

### 1. Predict Optimal Food Quantity

**Endpoint**: `/predict`

**Method**: POST

**Description**: Predicts the optimal quantity of a food item to prepare based on various input parameters including date, item ID, current stock, and rainfall.

**Request Body**:
```json
{
  "date": "YYYY-MM-DD",
  "item_id": "item_name",
  "current_stock": 50,
  "rainfall_today": 5.2
}
```

**Parameters**:
- `date` (required): The date for the prediction in YYYY-MM-DD format
- `item_id` (required): The identifier for the food item (e.g., "item_1", "item_2")
- `current_stock` (optional): The current available stock of the item
- `rainfall_today` (optional): The rainfall amount for the day in millimeters

**Response**:
```json
{
  "item_id": "item_name",
  "predicted_quantity": 75
}
```

**Example**:
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "date": "2023-07-15",
           "item_id": "item_1",
           "current_stock": 50,
           "rainfall_today": 5.2
         }'
```

### 2. API Documentation

**Endpoints**:
- `/docs` - Interactive Swagger documentation
- `/redoc` - Alternative ReDoc documentation

**Method**: GET

**Description**: Auto-generated API documentation where you can explore and test all endpoints.

## Integrating with Other Systems

### Python Client Example

```python
import requests
import json

def predict_food_quantity(date, item_id, current_stock=None, rainfall_today=None):
    url = "http://localhost:8000/predict"
    payload = {
        "date": date,
        "item_id": item_id
    }
    
    if current_stock is not None:
        payload["current_stock"] = current_stock
    
    if rainfall_today is not None:
        payload["rainfall_today"] = rainfall_today
        
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = predict_food_quantity("2023-07-15", "item_1", 50, 5.2)
print(result)
```

### JavaScript Client Example

```javascript
async function predictFoodQuantity(date, itemId, currentStock, rainfallToday) {
    const url = 'http://localhost:8000/predict';
    const payload = {
        date: date,
        item_id: itemId,
    };
    
    if (currentStock !== undefined) {
        payload.current_stock = currentStock;
    }
    
    if (rainfallToday !== undefined) {
        payload.rainfall_today = rainfallToday;
    }
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    });
    
    return await response.json();
}

// Example usage
predictFoodQuantity('2023-07-15', 'item_1', 50, 5.2)
    .then(result => console.log(result))
    .catch(error => console.error('Error:', error));
```

## Best Practices

1. **Date Format**: Always use YYYY-MM-DD format for dates
2. **Item IDs**: Use consistent item IDs as defined in your training data
3. **Production Deployment**: When deploying to production, use a proper WSGI server like Gunicorn:
   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker -w 4 src.api_backend:app
   ```
4. **Error Handling**: Always implement error handling in your client code to gracefully manage API unavailability

## Troubleshooting

- **Connection Issues**: Ensure the server is running and the port (8000) is not blocked by a firewall
- **Invalid Parameters**: Ensure all required parameters are provided and formatted correctly
- **Unexpected Predictions**: Verify that your data files and models are correctly loaded

## Modifying the Data Sources

If you want to modify the data that the prediction system uses, you'll need to understand where the data is stored and how it's processed.

### Data File Locations

1. **Historical Sales Data**: `data/historical_sales.csv`
   - This file contains past sales records used for training the model
   - Format: date, item_id, quantity_sold, price

2. **Weather Data**: `data/weather_data.csv`
   - This file contains weather information that may affect sales
   - Format: date, temperature, rainfall, humidity

3. **Academic Calendar**: `data/academic_calendar.csv`
   - This file contains information about academic events that may affect canteen traffic
   - Format: date, event_type, crowd_level

### Adding New Menu Items

To add new food items to the menu:

1. Add historical data for the new item in `data/historical_sales.csv` with the new `item_id`
2. Re-run the data preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
3. Re-train the ML model:
   ```bash
   python src/train_ml_model.py
   ```
4. Re-train the RL agent:
   ```bash
   python src/rl_agent.py
   ```

### Modifying Existing Items

To modify data for existing items:

1. Edit the appropriate entries in the CSV files
2. Follow the same reprocessing and retraining steps as above

### Data Format Requirements

When editing the CSV files, maintain these format requirements:

- `item_id`: Use the format "item_X" where X is a number or descriptive name
- `date`: Use YYYY-MM-DD format
- `quantity_sold`: Integer values
- `event_type`: String values like "holiday", "exam_period", "normal_day"
- `crowd_level`: Integer values (1-5) representing expected crowd density

### Example: Adding a New Menu Item

Here's an example of adding a new item "item_pizza" to the system:

1. Add entries to `historical_sales.csv`:
   ```
   2023-06-15,item_pizza,45,8.99
   2023-06-16,item_pizza,52,8.99
   2023-06-17,item_pizza,38,8.99
   ...
   ```

2. Re-process and re-train:
   ```bash
   python src/data_preprocessing.py
   python src/train_ml_model.py
   python src/rl_agent.py
   ```

3. Restart the API server

After completing these steps, you can make predictions for the new item using the API.

For more information, please refer to the project documentation or raise an issue on the project repository.
