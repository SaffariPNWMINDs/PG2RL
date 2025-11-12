import gym
import numpy as np
import random
import time

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Visualization enabled

# Discretize the continuous state space into a finite grid
num_bins = (10, 10, 10, 10)  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-1, 1)   # Clip velocity values
state_bounds[3] = (-1, 1)   # Clip pole angular velocity values

# Q-Learning Hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.99       # Discount factor
epsilon = 1.0      # Initial exploration probability
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.01     # Minimum epsilon value
num_episodes = 1000     # Total episodes to train
max_steps = 200        # Max steps per episode

# Initialize Q-table (state-action value function) with zeros
q_table = np.zeros(num_bins + (env.action_space.n,))  # Shape: (10, 10, 10, 10, 2)

# Function to discretize a continuous state
def discretize_state(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(4)]
    new_state = [int(np.digitize(ratio, np.linspace(0, 1, num_bins[i] - 1))) for i, ratio in enumerate(ratios)]
    return tuple(min(num_bins[i] - 1, max(0, new_state[i])) for i in range(4))

# Epsilon-greedy policy for action selection
def epsilon_greedy_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Q-Learning Algorithm
for episode in range(num_episodes):
    state, _ = env.reset()   # Only take observation
    state = discretize_state(state)
    done = False

    for step in range(max_steps):
        env.render()  # Render the environment (visualization)
        action = epsilon_greedy_action(state)  # Choose action
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Q-Learning update rule
        best_next_action = np.argmax(q_table[next_state])
        q_table[state + (action,)] += alpha * (
            reward + gamma * q_table[next_state + (best_next_action,)] - q_table[state + (action,)]
        )
        
        state = next_state  # Move to the next state

        if done:  # Episode ends when the pole falls or time limit is reached
            print(f"Episode {episode+1}: Completed in {step+1} steps")
            break
    
    # Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

env.close()

# Display the learned Q-table
print("Learned Q-Table:")
print(q_table)
