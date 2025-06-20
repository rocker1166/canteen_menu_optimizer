import numpy as np
import pandas as pd
import os

class EnhancedCanteenEnv:
    def __init__(self, historical_data_path, operational_data_path, weather_data_path, academic_calendar_path):
        # Load all data sources
        self.sales_data = pd.read_csv(historical_data_path)
        self.sales_data["date"] = pd.to_datetime(self.sales_data["date"])
        
        self.operational_data = pd.read_csv(operational_data_path)
        self.operational_data["date"] = pd.to_datetime(self.operational_data["date"])
        
        self.weather_data = pd.read_csv(weather_data_path)
        self.weather_data["date"] = pd.to_datetime(self.weather_data["date"])
        
        self.academic_data = pd.read_csv(academic_calendar_path)
        self.academic_data["date"] = pd.to_datetime(self.academic_data["date"])
        
        self.current_step = 0
        self.dates = sorted(self.sales_data["date"].unique())
        self.max_steps = len(self.dates)
        self.items = sorted(self.sales_data["item_id"].unique())

        # Enhanced costs and rewards with realistic values
        self.cost_per_unit = 15  # Average cost to prepare one unit
        self.revenue_per_unit = 35 # Average revenue from selling one unit
        self.waste_penalty_per_unit = 8 # Penalty for wasted food
        self.underproduction_penalty_per_unit = 20 # Penalty for not meeting demand

        # Enhanced action levels for quantity
        self.action_levels = [0, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300]

    def reset(self):
        self.current_step = 0
        return self._get_enhanced_state()

    def _get_enhanced_state(self):
        if self.current_step >= self.max_steps:
            return None

        current_date = self.dates[self.current_step]
        
        # 1. Day Context Features
        state_features = [
            current_date.weekday(),  # day_of_week (0-6)
            current_date.month,      # month (1-12)
            current_date.dayofyear,  # day_of_year (1-365)
            current_date.isocalendar().week,  # week_of_year
            1 if current_date.weekday() >= 5 else 0,  # is_weekend
        ]
        
        # 2. Operational Context Features
        op_data = self.operational_data[self.operational_data["date"] == current_date]
        if not op_data.empty:
            op_row = op_data.iloc[0]
            state_features.extend([
                op_row.get('student_count', 250),
                op_row.get('staff_available', 5),
                op_row.get('canteen_capacity', 300),
                op_row.get('event_today', 0),
                op_row.get('hostel_open', 1),
                op_row.get('is_holiday', 0),
                op_row.get('is_exam_period', 0),
            ])
        else:
            state_features.extend([250, 5, 300, 0, 1, 0, 0])  # Default values
        
        # 3. Weather Context Features
        weather_data = self.weather_data[self.weather_data["date"] == current_date]
        if not weather_data.empty:
            weather_row = weather_data.iloc[0]
            state_features.extend([
                weather_row.get('temperature', 25),
                weather_row.get('humidity', 70),
                weather_row.get('rainfall', 0),
                weather_row.get('feels_like_temp', 25),
            ])
        else:
            state_features.extend([25, 70, 0, 25])  # Default values

        # 4. Academic Calendar Features
        academic_data = self.academic_data[self.academic_data["date"] == current_date]
        if not academic_data.empty:
            academic_row = academic_data.iloc[0]
            state_features.extend([
                academic_row.get('is_exam_week', 0),
                academic_row.get('is_festival', 0),
            ])
        else:
            state_features.extend([0, 0])

        # 5. Historical Sales Context Features
        if self.current_step > 0:
            # Previous day sales
            prev_date = self.dates[self.current_step - 1]
            prev_sales = self.sales_data[self.sales_data["date"] == prev_date]
            prev_sales_by_item = prev_sales.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)
            state_features.extend(prev_sales_by_item.values)
            
            # Previous day estimated waste (5-15% of sales)
            prev_waste = prev_sales_by_item * np.random.uniform(0.05, 0.15, len(self.items))
            state_features.extend(prev_waste.values)
        else:
            # No previous data available
            state_features.extend([0] * len(self.items) * 2)
        
        # 6. 3-day average sales if we have enough history
        if self.current_step >= 3:
            recent_dates = self.dates[max(0, self.current_step-3):self.current_step]
            recent_sales = self.sales_data[self.sales_data["date"].isin(recent_dates)]
            avg_sales = recent_sales.groupby("item_id")["quantity_sold"].mean().reindex(self.items, fill_value=0)
            state_features.extend(avg_sales.values)
        else:
            state_features.extend([0] * len(self.items))

        # 7. Same day previous week sales (weekly seasonality)
        if self.current_step >= 7:
            prev_week_date = self.dates[self.current_step - 7]
            prev_week_sales = self.sales_data[self.sales_data["date"] == prev_week_date]
            prev_week_sales_by_item = prev_week_sales.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)
            state_features.extend(prev_week_sales_by_item.values)
        else:
            state_features.extend([0] * len(self.items))

        # 8. Seasonal and interaction features
        is_monsoon = 1 if current_date.month in [6, 7, 8, 9] else 0
        is_winter = 1 if current_date.month in [12, 1, 2] else 0
        is_summer = 1 if current_date.month in [3, 4, 5] else 0
        
        # Get current weather for interactions
        current_temp = state_features[12]  # temperature from weather features
        current_humidity = state_features[13]  # humidity from weather features
        current_rainfall = state_features[14]  # rainfall from weather features
        current_students = state_features[5]   # student count from operational features
        current_weekend = state_features[4]    # is_weekend from day features
        
        state_features.extend([
            is_monsoon, is_winter, is_summer,
            current_temp * current_humidity / 100,  # temp_humidity_interaction
            current_rainfall * (40 - current_temp), # rain_temp_interaction
            current_students * current_weekend       # student_weekend_interaction
        ])

        return np.array(state_features, dtype=np.float32)

    def step(self, action_index):
        prepared_qty = self.action_levels[action_index]
        current_date = self.dates[self.current_step]
        
        # Get actual demand for all items on this date
        daily_sales = self.sales_data[self.sales_data["date"] == current_date]
        daily_demand = daily_sales.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)

        total_reward = 0
        done = False

        # Calculate enhanced reward based on all items
        for item_id in self.items:
            actual_demand = daily_demand.get(item_id, 0)

            # For simplicity, assume the action applies proportionally to all items
            # In practice, you might want separate actions per item
            item_prepared = prepared_qty // len(self.items)  # Distribute equally
            
            sold_qty = min(item_prepared, actual_demand)
            waste_qty = max(0, item_prepared - actual_demand)
            unmet_demand = max(0, actual_demand - item_prepared)

            # Enhanced reward calculation
            revenue = sold_qty * self.revenue_per_unit
            cost = item_prepared * self.cost_per_unit
            waste_penalty = waste_qty * self.waste_penalty_per_unit
            underproduction_penalty = unmet_demand * self.underproduction_penalty_per_unit

            item_reward = revenue - cost - waste_penalty - underproduction_penalty
            total_reward += item_reward

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            next_state = None
        else:
            next_state = self._get_enhanced_state()

        return next_state, total_reward, done, {}

    def get_action_space_size(self):
        return len(self.action_levels)

    def get_state_space_size(self):
        # Calculate total state size based on enhanced features
        base_features = 5      # day_of_week, month, day_of_year, week_of_year, is_weekend
        operational_features = 7  # student_count, staff_available, canteen_capacity, event_today, hostel_open, is_holiday, is_exam_period
        weather_features = 4   # temperature, humidity, rainfall, feels_like_temp
        academic_features = 2  # is_exam_week, is_festival
        historical_features = len(self.items) * 4  # prev_sales + prev_waste + 3day_avg + prev_week_same_day per item
        seasonal_interaction_features = 6  # is_monsoon, is_winter, is_summer + 3 interaction features
        
        return base_features + operational_features + weather_features + academic_features + historical_features + seasonal_interaction_features

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, "data/historical_sales.csv")
    operational_path = os.path.join(base_dir, "data/operational_data.csv")
    weather_path = os.path.join(base_dir, "data/weather_data.csv")
    academic_path = os.path.join(base_dir, "data/academic_calendar.csv")
    
    env = EnhancedCanteenEnv(sales_path, operational_path, weather_path, academic_path)
    state = env.reset()
    print("Initial Enhanced State Shape:", state.shape)
    print("State space size:", env.get_state_space_size())
    print("Action space size:", env.get_action_space_size())
    print("First few state features:", state[:20])

    # Example: take a random action
    random_action_index = np.random.randint(0, env.get_action_space_size())
    next_state, reward, done, _ = env.step(random_action_index)
    print("\\nAfter action", random_action_index)
    print("Next State Shape:", next_state.shape if next_state is not None else "None")
    print("Reward:", reward)
    print("Done:", done)
