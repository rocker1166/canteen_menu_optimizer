
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(start_date, end_date):
    # Real menu items with price and cost
    menu = {
        "veg_biryani": {"price": 50, "cost": 30, "popularity": 1.2},
        "fish_curry_rice": {"price": 80, "cost": 50, "popularity": 1.1},
        "luchi_aloo": {"price": 30, "cost": 15, "popularity": 0.9},
        "ghugni": {"price": 25, "cost": 10, "popularity": 0.8},
        "maggi": {"price": 20, "cost": 8, "popularity": 1.5},
        "tea_biscuit": {"price": 15, "cost": 5, "popularity": 1.3},
        "chicken_roll": {"price": 40, "cost": 25, "popularity": 1.2},
        "egg_roll": {"price": 35, "cost": 20, "popularity": 1.0},
        "veg_momo": {"price": 45, "cost": 25, "popularity": 0.9},
        "ice_cream": {"price": 30, "cost": 15, "popularity": 0.7}
    }
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    for date in dates:
        for item_id, item_info in menu.items():
            # Simulate sales based on day of week, and add some randomness
            base_sales = np.random.randint(50, 200) # Base sales for an item
            if date.weekday() >= 5: # Weekend
                base_sales = np.random.randint(20, 100)
            
            # Add some variation based on item popularity
            base_sales *= item_info["popularity"]
            
            # Special case for maggi: sells more when it rains
            if item_id == "maggi" and np.random.rand() < 0.3:  # Rainy day
                base_sales *= 1.3
                
            # Add noise
            sales = max(0, int(base_sales + np.random.normal(0, 20)))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'item_id': item_id,
                'item_name': item_id.replace('_', ' ').title(),
                'quantity_sold': sales
            })
    return pd.DataFrame(data)

def generate_weather_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    for date in dates:
        # Simulate weather conditions
        temp = np.random.randint(20, 35) # Temperature in Celsius
        humidity = np.random.randint(60, 95) # Humidity percentage
        rainfall = 0
        if np.random.rand() < 0.3: # 30% chance of rain
            rainfall = np.random.uniform(0, 50) # Rainfall in mm
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'temperature': temp,
            'humidity': humidity,
            'rainfall': rainfall
        })
    return pd.DataFrame(data)

def generate_academic_calendar_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    holidays = []
    exam_weeks = []
    festivals = []

    # Example holidays (simplified)
    holidays.append(pd.to_datetime('2024-01-26')) # Republic Day
    holidays.append(pd.to_datetime('2024-08-15')) # Independence Day
    holidays.append(pd.to_datetime('2024-10-02')) # Gandhi Jayanti

    # Example exam weeks (simplified)
    exam_weeks.append((pd.to_datetime('2024-05-01'), pd.to_datetime('2024-05-15')))
    exam_weeks.append((pd.to_datetime('2024-11-15'), pd.to_datetime('2024-11-30')))

    # Example festivals (simplified)
    festivals.append(pd.to_datetime('2024-03-25')) # Holi
    festivals.append(pd.to_datetime('2024-10-12')) # Durga Puja (example start)

    for date in dates:
        is_holiday = 0
        is_exam_week = 0
        is_festival = 0

        if date in holidays:
            is_holiday = 1
        
        for start, end in exam_weeks:
            if start <= date <= end:
                is_exam_week = 1
                break
        
        if date in festivals:
            is_festival = 1

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'is_holiday': is_holiday,
            'is_exam_week': is_exam_week,
            'is_festival': is_festival
        })
    return pd.DataFrame(data)

if __name__ == '__main__':
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    print("Generating sales data...")
    sales_df = generate_sales_data(start_date, end_date)
    sales_df.to_csv('data/historical_sales.csv', index=False)
    print("Historical sales data generated and saved to historical_sales.csv")

    print("Generating weather data...")
    weather_df = generate_weather_data(start_date, end_date)
    weather_df.to_csv('data/weather_data.csv', index=False)
    print("Weather data generated and saved to weather_data.csv")

    print("Generating academic calendar data...")
    academic_calendar_df = generate_academic_calendar_data(start_date, end_date)
    academic_calendar_df.to_csv('data/academic_calendar.csv', index=False)
    print("Academic calendar data generated and saved to academic_calendar.csv")


