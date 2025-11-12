import gym
import numpy as np
import time  # For visualization slowdown

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Enable visualization

# TD(lambda) Hyperparameters
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
lambda_ = 0.9       # Lambda for eligibility traces
num_episodes = 10   # Number of episodes
epsilon = 0.1       # Exploration probability for epsilon-greedy policy

# Discretize the continuous state space into a finite grid
num_bins = (10, 10, 10, 10)  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-1, 1)   # Clip velocity values
state_bounds[3] = (-1, 1)   # Clip pole angular velocity values

# Initialize value function and eligibility traces
value_table = np.zeros(num_bins)
eligibility_traces = np.zeros(num_bins)

# Function to discretize a continuous state
def discretize_state(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(4)]
    new_state = [int(np.digitize(ratio, np.linspace(0, 1, num_bins[i] - 1))) for i, ratio in enumerate(ratios)]
    return tuple(min(num_bins[i] - 1, max(0, new_state[i])) for i in range(4))

# Epsilon-greedy action selection
def epsilon_greedy_action():
    return np.random.choice([0, 1])  # Random action for now

# TD(lambda) with eligibility traces
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    eligibility_traces = np.zeros(num_bins)  # Reset eligibility traces
    done = False
    
    while not done:
        env.render()  # Visualize the environment
        time.sleep(0.02)  # Slow down for better visibility

        # Select an action (random for simplicity)
        action = epsilon_greedy_action()
        
        # Take action, observe reward and next state
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # TD error (delta)
        current_value = value_table[state]
        next_value = value_table[next_state] if not done else 0
        delta = reward + gamma * next_value - current_value
        
        # Update eligibility trace for the current state
        eligibility_traces[state] += 1.0
        
        # Update all states' values using eligibility traces
        for s in np.ndindex(value_table.shape):
            value_table[s] += alpha * delta * eligibility_traces[s]
            eligibility_traces[s] *= gamma * lambda_  # Decay traces
        
        # Move to the next state
        state = next_state
    
    print(f"Episode {episode + 1}/{num_episodes} complete.")

env.close()

# Display Value Table
print("Learned Value Table:")
print(value_table)
