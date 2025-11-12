import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.fc(state)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.fc(state)

# A2C training loop
def train(env, actor, critic, num_episodes=1000, gamma=0.99, lr=0.001):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    rewards = []  # To store total rewards for each episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        done = False
        while not done:
            # Select action
            action_probs = actor(state)
            action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())

            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Compute TD target
            td_target = reward + gamma * critic(next_state).detach() * (1 - int(done))
            value = critic(state)
            
            # Compute Advantage
            advantage = td_target - value

            # Critic update
            critic_loss = advantage.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update
            log_prob = torch.log(action_probs[action])
            actor_loss = -log_prob * advantage.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance (A2C)')
    plt.show()

# Example usage (use any OpenAI Gym environment)
import gym
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

train(env, actor, critic)
