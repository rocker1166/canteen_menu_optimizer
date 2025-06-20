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
                'quantity_sold': sales,
                'price': item_info["price"],
                'cost': item_info["cost"]
            })
    return pd.DataFrame(data)

def generate_operational_data(start_date, end_date):
    """Generate operational data for canteen including student count, staff, events etc."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    # West Bengal specific holidays and events
    wb_holidays = [
        '2023-01-26', '2023-03-08', '2023-03-21', '2023-04-14', '2023-04-18',
        '2023-08-15', '2023-08-30', '2023-10-02', '2023-10-24', '2023-11-07',
        '2023-11-27', '2023-12-25', '2024-01-26', '2024-03-08', '2024-03-25',
        '2024-04-14', '2024-04-18', '2024-08-15', '2024-08-30', '2024-10-02',
        '2024-10-24', '2024-11-07', '2024-11-27', '2024-12-25'
    ]
    
    # Exam periods (typically May and November)
    exam_periods = [
        (datetime(2023, 5, 1), datetime(2023, 5, 31)),
        (datetime(2023, 11, 15), datetime(2023, 12, 15)),
        (datetime(2024, 5, 1), datetime(2024, 5, 31)),
        (datetime(2024, 11, 15), datetime(2024, 12, 15))
    ]
    
    for date in dates:
        # Base student count (varies by day of week)
        base_students = 280 if date.weekday() < 5 else 120  # Weekday vs weekend
        
        # Seasonal variations
        if date.month in [6, 7]:  # Summer vacation
            base_students *= 0.3
        elif date.month in [12, 1]:  # Winter break
            base_students *= 0.5
        
        # Holiday effects
        is_holiday = 1 if date.strftime('%Y-%m-%d') in wb_holidays else 0
        if is_holiday:
            base_students *= 0.1
        
        # Exam period effects
        is_exam_period = 0
        for exam_start, exam_end in exam_periods:
            if exam_start <= date <= exam_end:
                is_exam_period = 1
                base_students *= 1.4  # More students during exams
                break
        
        # Events (festivals, competitions)
        event_today = 0
        if date.month == 10 and date.day in [12, 13, 14, 15, 16]:  # Durga Puja
            event_today = 1
            base_students *= 0.2
        elif date.month == 3 and date.day == 21:  # Holi
            event_today = 1
            base_students *= 0.3
        elif np.random.rand() < 0.05:  # Random events (sports day, cultural fest)
            event_today = 1
            base_students *= 1.6
        
        # Add noise
        student_count = max(10, int(base_students + np.random.normal(0, 20)))
        
        # Staff availability (affects production capacity)
        base_staff = 6
        if date.weekday() >= 5:  # Weekend
            base_staff = 3
        if is_holiday:
            base_staff = 2
        staff_available = max(1, base_staff + np.random.randint(-1, 2))
        
        # Canteen capacity
        canteen_capacity = 320 if not event_today else 450
        
        # Hostel status
        hostel_open = 0 if (date.month in [6, 7] or is_holiday) else 1
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'student_count': student_count,
            'staff_available': staff_available,
            'canteen_capacity': canteen_capacity,
            'event_today': event_today,
            'hostel_open': hostel_open,
            'is_holiday': is_holiday,
            'is_exam_period': is_exam_period
        })
    
    return pd.DataFrame(data)

def generate_enhanced_weather_data(start_date, end_date):
    """Generate enhanced weather data with more realistic patterns for West Bengal"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for date in dates:
        # Seasonal temperature patterns for West Bengal
        if date.month in [12, 1, 2]:  # Winter
            temp_base = np.random.normal(20, 5)
        elif date.month in [3, 4, 5]:  # Summer
            temp_base = np.random.normal(35, 8)
        elif date.month in [6, 7, 8, 9]:  # Monsoon
            temp_base = np.random.normal(28, 4)
        else:  # Post-monsoon
            temp_base = np.random.normal(25, 6)
        
        temperature = max(15, min(45, temp_base))
        
        # Humidity patterns
        if date.month in [6, 7, 8, 9]:  # Monsoon - high humidity
            humidity = np.random.normal(85, 10)
        else:
            humidity = np.random.normal(65, 15)
        
        humidity = max(40, min(100, humidity))
        
        # Rainfall patterns
        rainfall = 0
        if date.month in [6, 7, 8, 9]:  # Monsoon season
            if np.random.rand() < 0.6:  # 60% chance of rain
                rainfall = np.random.exponential(15)
        elif date.month in [3, 4, 5]:  # Summer storms
            if np.random.rand() < 0.2:  # 20% chance of rain
                rainfall = np.random.exponential(8)
        else:  # Other months
            if np.random.rand() < 0.1:  # 10% chance of rain
                rainfall = np.random.exponential(5)
        
        # Feels like temperature
        feels_like_temp = temperature + (humidity - 60) * 0.1
        if rainfall > 10:
            feels_like_temp -= 2  # Rain cools things down
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 1),
            'feels_like_temp': round(feels_like_temp, 1)
        })
    
    return pd.DataFrame(data)

def generate_academic_calendar_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    holidays = []
    exam_weeks = []
    festivals = []

    # West Bengal holidays
    holidays.extend([
        pd.to_datetime('2023-01-26'), pd.to_datetime('2023-03-08'),
        pd.to_datetime('2023-08-15'), pd.to_datetime('2023-10-02'),
        pd.to_datetime('2024-01-26'), pd.to_datetime('2024-03-08'),
        pd.to_datetime('2024-08-15'), pd.to_datetime('2024-10-02')
    ])

    # Exam weeks
    exam_weeks.extend([
        (pd.to_datetime('2023-05-01'), pd.to_datetime('2023-05-31')),
        (pd.to_datetime('2023-11-15'), pd.to_datetime('2023-12-15')),
        (pd.to_datetime('2024-05-01'), pd.to_datetime('2024-05-31')),
        (pd.to_datetime('2024-11-15'), pd.to_datetime('2024-12-15'))
    ])

    # Festivals
    festivals.extend([
        pd.to_datetime('2023-03-08'), pd.to_datetime('2023-10-12'),
        pd.to_datetime('2024-03-25'), pd.to_datetime('2024-10-12')
    ])

    for date in dates:
        is_holiday = 1 if date in holidays else 0
        is_exam_week = 0
        for start, end in exam_weeks:
            if start <= date <= end:
                is_exam_week = 1
                break
        is_festival = 1 if date in festivals else 0

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

    print("Generating operational data...")
    operational_df = generate_operational_data(start_date, end_date)
    operational_df.to_csv('data/operational_data.csv', index=False)
    print("Operational data generated and saved to operational_data.csv")

    print("Generating enhanced weather data...")
    weather_df = generate_enhanced_weather_data(start_date, end_date)
    weather_df.to_csv('data/weather_data.csv', index=False)
    print("Enhanced weather data generated and saved to weather_data.csv")

    print("Generating sales data...")
    sales_df = generate_sales_data(start_date, end_date)
    sales_df.to_csv('data/historical_sales.csv', index=False)
    print("Historical sales data generated and saved to historical_sales.csv")

    print("Generating academic calendar data...")
    academic_calendar_df = generate_academic_calendar_data(start_date, end_date)
    academic_calendar_df.to_csv('data/academic_calendar.csv', index=False)
    print("Academic calendar data generated and saved to academic_calendar.csv")
