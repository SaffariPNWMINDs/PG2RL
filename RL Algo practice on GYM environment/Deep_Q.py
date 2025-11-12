import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
import matplotlib.pyplot as plt

# Initialize the CartPole environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4 state variables
action_size = env.action_space.n  # 2 actions (left or right)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99                 # Discount factor
epsilon = 1.0                # Initial exploration probability
epsilon_decay = 0.995        # Decay rate for epsilon
epsilon_min = 0.01           # Minimum epsilon
batch_size = 64              # Size of training batches
memory_size = 2000           # Replay buffer size
target_update_freq = 10      # Frequency of updating target network
num_episodes = 100           # Number of training episodes
max_steps = 200              # Max steps per episode

# Replay buffer for experience replay
memory = deque(maxlen=memory_size)

# Neural Network Models for Policy and Target Networks
def build_model():
    model = Sequential([
        Dense(24, input_dim=state_size, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

policy_network = build_model()
target_network = build_model()
target_network.set_weights(policy_network.get_weights())  # Synchronize target with policy

# Epsilon-greedy policy for action selection
def epsilon_greedy_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explore
    q_values = policy_network.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])  # Exploit

# Function to replay experiences and train the policy network
def replay_and_train():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = np.array(states)
    next_states = np.array(next_states)
    q_values = policy_network.predict(states, verbose=0)
    q_next_values = target_network.predict(next_states, verbose=0)

    for i in range(batch_size):
        q_target = rewards[i]
        if not dones[i]:
            q_target += gamma * np.max(q_next_values[i])
        q_values[i][actions[i]] = q_target

    policy_network.fit(states, q_values, epochs=1, verbose=0)

# Training Loop
rewards_per_episode = []
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Add experience to the replay buffer
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Train the policy network using experience replay
        replay_and_train()

        if done:
            break

    # Update epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    rewards_per_episode.append(total_reward)

    # Update target network
    if episode % target_update_freq == 0:
        target_network.set_weights(policy_network.get_weights())

    print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()

# Plot Rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Performance of DQN on CartPole')
plt.show()
