import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

# Actor Network
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

# Critic Network
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

# PPO Training Function
def train_ppo(env, actor, critic, num_episodes=1000, gamma=0.99, lr=0.001, epsilon=0.2, epochs=4, batch_size=64, max_steps=1000):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    total_rewards_per_episode = []  # List to store total rewards

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        
        # Storage for trajectory
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        total_reward = 0
        step_count = 0

        # Generate an episode
        done = False
        while not done and step_count < max_steps:  # Limit the number of steps in an episode
            step_count += 1

            action_probs = actor(state)
            action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
            log_prob = torch.log(action_probs[action])
            
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.detach())  # Detach log_probs to avoid graph reuse
            
            state = next_state
            total_reward += reward

        # Store the total reward for this episode
        total_rewards_per_episode.append(total_reward)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # Compute returns and advantages
        states = torch.stack(states)
        actions = torch.tensor(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_reward = reward + gamma * discounted_reward * (1 - done)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (returns - critic(states).squeeze(1)).detach()

        # Update Actor and Critic using PPO
        for _ in range(epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_log_probs = log_probs[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]
                
                # Recompute log_probs for the batch
                new_action_probs = actor(batch_states)
                new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(1))).squeeze(1)

                # Compute policy ratio
                policy_ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # PPO clipped loss
                clipped_ratio = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
                actor_loss = -torch.min(policy_ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Critic loss
                critic_loss = (batch_returns - critic(batch_states).squeeze(1)).pow(2).mean()
                
                # Backpropagation for actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Backpropagation for critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

    # Plot the total rewards
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.show()

# Main execution
env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

train_ppo(env, actor, critic, num_episodes=1000, max_steps=1000)