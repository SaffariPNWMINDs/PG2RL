import gym
import numpy as np
import time  # For slowing down the visualization

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Enable visualization

# TD(0) Hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.99       # Discount factor
num_episodes = 50  # Number of episodes for visualization
epsilon = 0.1      # Exploration probability for epsilon-greedy policy

# Discretize the continuous state space into a finite grid
num_bins = (10, 10, 10, 10)  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-1, 1)   # Clip velocity values
state_bounds[3] = (-1, 1)   # Clip pole angular velocity values

# Initialize value function as a table
value_table = np.zeros(num_bins)
print(value_table)

# Function to discretize a continuous state
def discretize_state(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(4)]
    new_state = [int(np.digitize(ratio, np.linspace(0, 1, num_bins[i] - 1))) for i, ratio in enumerate(ratios)]
    return tuple(min(num_bins[i] - 1, max(0, new_state[i])) for i in range(4))

# Epsilon-greedy policy for action selection
def epsilon_greedy_action():
    return np.random.choice([0, 1])

# TD(0) Update Rule
for episode in range(num_episodes):
    state, _ = env.reset()   # Only take the observation, ignore "info"
    state = discretize_state(state)
    done = False
    
    while not done:
        env.render()  # Render the environment for visualization
        time.sleep(0.02)  # Slow down rendering for better visibility

        # Choose an action using epsilon-greedy policy
        action = epsilon_greedy_action()

        # Perform action and observe next state and reward
        next_state, reward, done, _, _ = env.step(action)  # Fix to unpack step output
        next_state = discretize_state(next_state)
        
        # TD(0) Update Rule for Value Function
        current_value = value_table[state]
        next_value = value_table[next_state] if not done else 0
        value_table[state] += alpha * (reward + gamma * next_value - current_value)
        
        # Move to the next state
        state = next_state

    print(f"Episode {episode+1}/{num_episodes} complete.")

env.close()

# Display Value Table
print("Learned Value Table:")
print(value_table)
