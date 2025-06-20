import numpy as np
import pandas as pd
import random
from collections import defaultdict
import joblib
import os

class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay_rate=0.995, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        
        # Track learning progress
        self.episode_rewards = []
        self.epsilon_history = []

    def _state_to_tuple(self, state):
        """Convert numpy array state to a hashable tuple with quantization for large state spaces"""
        # Use percentile-based quantization for better state representation
        if len(state) > 20:  # For large state spaces, use coarser quantization
            bins = 5
        else:
            bins = 10
            
        # Quantize each dimension independently based on its range
        quantized_state = []
        for i, value in enumerate(state):
            # Use fixed ranges for known feature types to ensure consistency
            if i < 5:  # Day context features
                if i == 0:  # day_of_week
                    quantized_state.append(int(value) if -1 <= value <= 1 else 0)
                elif i == 1:  # month  
                    quantized_state.append(int(value) if -2 <= value <= 2 else 0)
                else:
                    quantized_state.append(int(np.digitize(value, np.linspace(-3, 3, bins))))
            elif i < 9:  # Weather features (temperature, humidity, rainfall, feels_like)
                quantized_state.append(int(np.digitize(value, np.linspace(-3, 3, bins))))
            else:  # Other features
                quantized_state.append(int(np.digitize(value, np.linspace(-2, 2, bins))))
                
        return tuple(quantized_state)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state_tuple = self._state_to_tuple(state)
            return np.argmax(self.q_table[state_tuple])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        state_tuple = self._state_to_tuple(state)
        
        if done or next_state is None:
            target = reward
        else:
            next_state_tuple = self._state_to_tuple(next_state)
            target = reward + self.discount_factor * np.max(self.q_table[next_state_tuple])

        # Q-learning update with enhanced learning rate decay
        current_q = self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] = current_q + self.learning_rate * (target - current_q)

        # Decay epsilon more gradually for better exploration
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay_rate

    def save_model(self, filepath):
        """Save the Q-table and agent parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'epsilon_history': self.epsilon_history
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """Load the Q-table and agent parameters"""
        model_data = joblib.load(filepath)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), model_data['q_table'])
        self.state_size = model_data['state_size']
        self.action_size = model_data['action_size']
        self.epsilon = model_data['epsilon']
        if 'episode_rewards' in model_data:
            self.episode_rewards = model_data['episode_rewards']
        if 'epsilon_history' in model_data:
            self.epsilon_history = model_data['epsilon_history']

    def get_q_value(self, state, action):
        """Get Q-value for debugging and analysis"""
        state_tuple = self._state_to_tuple(state)
        return self.q_table[state_tuple][action]

if __name__ == '__main__':
    from enhanced_canteen_env import EnhancedCanteenEnv
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, "data/historical_sales.csv")
    operational_path = os.path.join(base_dir, "data/operational_data.csv")
    weather_path = os.path.join(base_dir, "data/weather_data.csv")
    academic_path = os.path.join(base_dir, "data/academic_calendar.csv")
    
    env = EnhancedCanteenEnv(sales_path, operational_path, weather_path, academic_path)
    state_size = env.get_state_space_size()
    action_size = env.get_action_space_size()

    print(f"Training Enhanced RL Agent with:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    agent = EnhancedQLearningAgent(state_size, action_size)

    episodes = 150  # Increased episodes for better learning
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 1000:  # Prevent infinite loops
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

        agent.episode_rewards.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        if total_reward > best_reward:
            best_reward = total_reward
            
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            print(f"Episode {episode + 1}: Avg Reward (last 10) = {avg_reward:.0f}, "
                  f"Current Reward = {total_reward:.0f}, Epsilon = {agent.epsilon:.3f}")

    print(f"\\nTraining completed!")
    print(f"Best reward achieved: {best_reward:.0f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Total Q-table entries: {len(agent.q_table)}")

    # Save enhanced model
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    agent.save_model(os.path.join(models_dir, "enhanced_rl_q_table.pkl"))
    print("Enhanced RL Q-table saved.")
    
    # Save training history
    history_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'reward': agent.episode_rewards,
        'epsilon': agent.epsilon_history
    })
    history_df.to_csv(os.path.join(base_dir, "data/rl_training_history.csv"), index=False)
    print("RL training history saved.")
