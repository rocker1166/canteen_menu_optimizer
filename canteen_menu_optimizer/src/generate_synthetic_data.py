
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(start_date, end_date, num_items=10):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    for date in dates:
        for item_id in range(1, num_items + 1):
            # Simulate sales based on day of week, and add some randomness
            base_sales = np.random.randint(50, 200) # Base sales for an item
            if date.weekday() >= 5: # Weekend
                base_sales = np.random.randint(20, 100)
            
            # Add some variation for specific items or days
            if item_id == 1: # Popular item
                base_sales *= 1.2
            elif item_id == 5: # Less popular item
                base_sales *= 0.7

            # Add noise
            sales = max(0, int(base_sales + np.random.normal(0, 20)))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'item_id': f'item_{item_id}',
                'item_name': f'Dish {item_id}',
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
    sales_df.to_csv('canteen_menu_optimizer/data/historical_sales.csv', index=False)
    print("Historical sales data generated and saved to historical_sales.csv")

    print("Generating weather data...")
    weather_df = generate_weather_data(start_date, end_date)
    weather_df.to_csv('canteen_menu_optimizer/data/weather_data.csv', index=False)
    print("Weather data generated and saved to weather_data.csv")

    print("Generating academic calendar data...")
    academic_calendar_df = generate_academic_calendar_data(start_date, end_date)
    academic_calendar_df.to_csv('canteen_menu_optimizer/data/academic_calendar.csv', index=False)
    print("Academic calendar data generated and saved to academic_calendar.csv")


