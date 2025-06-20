
import numpy as np
import pandas as pd
import os

class CanteenEnv:
    def __init__(self, historical_data_path, operational_data_path=None, weather_data_path=None):
        self.sales_data = pd.read_csv(historical_data_path)
        self.sales_data["date"] = pd.to_datetime(self.sales_data["date"])
        
        # Load additional data sources if provided
        if operational_data_path and os.path.exists(operational_data_path):
            self.operational_data = pd.read_csv(operational_data_path)
            self.operational_data["date"] = pd.to_datetime(self.operational_data["date"])
        else:
            self.operational_data = None
            
        if weather_data_path and os.path.exists(weather_data_path):
            self.weather_data = pd.read_csv(weather_data_path)
            self.weather_data["date"] = pd.to_datetime(self.weather_data["date"])
        else:
            self.weather_data = None
        
        self.current_step = 0
        self.dates = sorted(self.sales_data["date"].unique())
        self.max_steps = len(self.dates)
        self.items = sorted(self.sales_data["item_id"].unique())

        # Define costs and rewards with realistic values
        self.cost_per_unit = 15  # Average cost to prepare one unit
        self.revenue_per_unit = 35 # Average revenue from selling one unit
        self.waste_penalty_per_unit = 8 # Penalty for wasted food
        self.underproduction_penalty_per_unit = 20 # Penalty for not meeting demand

        # Enhanced action levels for quantity
        self.action_levels = [0, 20, 40, 60, 80, 100, 150, 200, 250, 300]

    def reset(self):
        self.current_step = 0
        return self._get_enhanced_state()

    def _get_enhanced_state(self):
        if self.current_step >= self.max_steps:
            return None

        current_date = self.dates[self.current_step]
        
        # Base temporal features
        state_features = [
            current_date.weekday(),  # day_of_week
            current_date.month,      # month
            1 if current_date.weekday() >= 5 else 0,  # is_weekend
        ]
        
        # Get operational data if available
        if self.operational_data is not None:
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
        else:
            state_features.extend([250, 5, 300, 0, 1, 0, 0])
        
        # Get weather data if available
        if self.weather_data is not None:
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
        else:
            state_features.extend([25, 70, 0, 25])

        # Historical sales features
        if self.current_step > 0:
            prev_date = self.dates[self.current_step - 1]
            prev_sales = self.sales_data[self.sales_data["date"] == prev_date]
            prev_sales_by_item = prev_sales.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)
            state_features.extend(prev_sales_by_item.values)
            
            # Add waste data if available
            if 'waste_quantity' in prev_sales.columns:
                prev_waste_by_item = prev_sales.set_index("item_id")["waste_quantity"].reindex(self.items, fill_value=0)
                state_features.extend(prev_waste_by_item.values)
            else:
                state_features.extend([0] * len(self.items))
        else:
            # No previous data available
            state_features.extend([0] * len(self.items) * 2)
        
        # 3-day average sales if we have enough history
        if self.current_step >= 3:
            recent_dates = self.dates[max(0, self.current_step-3):self.current_step]
            recent_sales = self.sales_data[self.sales_data["date"].isin(recent_dates)]
            avg_sales = recent_sales.groupby("item_id")["quantity_sold"].mean().reindex(self.items, fill_value=0)
            state_features.extend(avg_sales.values)
        else:
            state_features.extend([0] * len(self.items))

        return np.array(state_features, dtype=np.float32)

    def step(self, action_index):
        prepared_qty = self.action_levels[action_index]
        current_date = self.dates[self.current_step]
        
        # Get actual demand for all items on this date
        daily_sales = self.sales_data[self.sales_data["date"] == current_date]
        daily_demand = daily_sales.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)

        total_reward = 0
        done = False

        # Calculate reward based on all items (simplified for demo)
        # In practice, you might want to optimize for one item at a time
        for item_id in self.items:
            actual_demand = daily_demand.get(item_id, 0)

            # For simplicity, assume the action applies to all items
            # In a more complex scenario, you'd have separate actions per item
            sold_qty = min(prepared_qty, actual_demand)
            waste_qty = max(0, prepared_qty - actual_demand)
            underproduction_qty = max(0, actual_demand - prepared_qty)

            # Calculate reward for this item
            revenue = sold_qty * self.revenue_per_unit
            cost = prepared_qty * self.cost_per_unit
            waste_penalty = waste_qty * self.waste_penalty_per_unit
            underproduction_penalty = underproduction_qty * self.underproduction_penalty_per_unit
            
            item_reward = revenue - cost - waste_penalty - underproduction_penalty
            total_reward += item_reward

        self.current_step += 1
        next_state = self._get_enhanced_state()

        if self.current_step >= self.max_steps:
            done = True

        return next_state, total_reward, done, {}

    def get_action_space_size(self):
        return len(self.action_levels)

    def get_state_space_size(self):
        # Calculate total state size based on enhanced features
        base_features = 7  # day_of_week, month, is_weekend + 4 operational features
        weather_features = 4  # temperature, humidity, rainfall, feels_like_temp
        historical_features = len(self.items) * 3  # prev_sales + prev_waste + 3day_avg per item
        
        return base_features + weather_features + historical_features

if __name__ == '__main__':
    env = CanteenEnv("canteen_menu_optimizer/data/historical_sales.csv")
    state = env.reset()
    print("Initial State:", state)

    # Example: take a random action (prepare 100 units of each item)
    random_action_index = np.random.randint(0, env.get_action_space_size())
    next_state, reward, done, _ = env.step(random_action_index)
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Done:", done)


