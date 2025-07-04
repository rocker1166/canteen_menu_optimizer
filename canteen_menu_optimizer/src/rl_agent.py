
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import joblib
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

    def _state_to_tuple(self, state):
        # Convert numpy array state to a hashable tuple
        # Quantize state for Q-table indexing if state is continuous
        return tuple(np.digitize(state, bins=np.linspace(state.min(), state.max(), 10)).astype(int))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1) # Explore
        else:
            return np.argmax(self.q_table[self._state_to_tuple(state)]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        state_tuple = self._state_to_tuple(state)
        next_state_tuple = self._state_to_tuple(next_state) if next_state is not None else None

        current_q = self.q_table[state_tuple][action]
        max_next_q = np.max(self.q_table[next_state_tuple]) if next_state is not None else 0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_tuple][action] = new_q

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def save_model(self, path):
        joblib.dump(dict(self.q_table), path)

    def load_model(self, path):
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), joblib.load(path))


if __name__ == '__main__':
    from canteen_env import CanteenEnv
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, "data/historical_sales.csv")
    operational_path = os.path.join(base_dir, "data/operational_data.csv")
    weather_path = os.path.join(base_dir, "data/weather_data.csv")
    
    env = CanteenEnv(sales_path, operational_path, weather_path)
    state_size = env.get_state_space_size()
    action_size = env.get_action_space_size()
    
    print(f"Enhanced RL Environment - State size: {state_size}, Action size: {action_size}")

    agent = QLearningAgent(state_size, action_size, epsilon=0.9, epsilon_decay_rate=0.005)

    episodes = 150  # More episodes for the more complex environment
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:  # Add step limit to prevent infinite loops
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        if episode % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {agent.epsilon:.3f}")

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    agent.save_model(os.path.join(models_dir, "enhanced_rl_q_table.pkl"))
    print("Enhanced RL Q-table saved.")
