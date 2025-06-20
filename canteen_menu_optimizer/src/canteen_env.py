
import numpy as np
import pandas as pd
import os

class CanteenEnv:
    def __init__(self, historical_data_path):
        self.data = pd.read_csv(historical_data_path)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.current_step = 0
        self.max_steps = len(self.data.groupby("date").first())
        self.dates = sorted(self.data["date"].unique())
        self.items = sorted(self.data["item_id"].unique())

        # Define costs and rewards (simplified for now)
        self.cost_per_unit = 10  # Cost to prepare one unit of food
        self.revenue_per_unit = 25 # Revenue from selling one unit of food
        self.waste_penalty_per_unit = 5 # Penalty for wasted food
        self.underproduction_penalty_per_unit = 15 # Penalty for not meeting demand

        # Define discrete action levels for quantity (e.g., 0, 50, 100, 150, 200)
        self.action_levels = [0, 50, 100, 150, 200, 250, 300]

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        if self.current_step >= self.max_steps:
            return None # Episode finished

        current_date = self.dates[self.current_step]
        
        # Get sales data for the current date
        daily_data = self.data[self.data["date"] == current_date]
        
        # Example state: sales of each item on the previous day
        if self.current_step > 0:
            prev_date = self.dates[self.current_step - 1]
            prev_daily_data = self.data[self.data["date"] == prev_date]
            state_features = prev_daily_data.set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0).values
        else:
            state_features = np.zeros(len(self.items)) # No previous sales for the first day

        # Add some temporal features to the state (e.g., day of week, month)
        state_features = np.append(state_features, current_date.dayofweek)
        state_features = np.append(state_features, current_date.month)

        return state_features

    def step(self, action_index):
        # Action is a single index representing the quantity for the current item being considered
        # For simplicity, we'll assume the RL agent is optimizing for one item at a time, or
        # that the action_index maps to a quantity for all items (less realistic for RL).
        # For this simplified env, let's assume the action_index corresponds to the quantity for ALL items.
        # A more complex env would iterate through items or have a multi-dimensional action.
        
        prepared_qty = self.action_levels[action_index]

        current_date = self.dates[self.current_step]
        daily_demand = self.data[self.data["date"] == current_date].set_index("item_id")["quantity_sold"].reindex(self.items, fill_value=0)

        total_reward = 0
        done = False

        for item_id in self.items:
            actual_demand = daily_demand.get(item_id, 0)

            # Calculate sales
            sold_qty = min(prepared_qty, actual_demand)
            
            # Calculate waste
            waste_qty = max(0, prepared_qty - actual_demand)

            # Calculate underproduction
            underproduction_qty = max(0, actual_demand - prepared_qty)

            # Calculate reward for this item
            reward = (sold_qty * self.revenue_per_unit) \
                     - (prepared_qty * self.cost_per_unit) \
                     - (waste_qty * self.waste_penalty_per_unit) \
                     - (underproduction_qty * self.underproduction_penalty_per_unit)
            
            total_reward += reward

        self.current_step += 1
        next_state = self._get_state()

        if self.current_step >= self.max_steps:
            done = True

        return next_state, total_reward, done, {}

    def get_action_space_size(self):
        return len(self.action_levels)

    def get_state_space_size(self):
        # Number of items + 2 (dayofweek, month)
        return len(self.items) + 2

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


