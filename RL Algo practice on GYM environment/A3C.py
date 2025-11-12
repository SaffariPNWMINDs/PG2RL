import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing
import gym
import time

# Shared Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.fc(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

# Worker Process
def worker(worker_id, env_name, global_model, optimizer, num_episodes, gamma, lock):
    local_model = ActorCritic(global_model.fc[0].in_features, global_model.actor[0].out_features)
    local_model.load_state_dict(global_model.state_dict())

    env = gym.make(env_name)
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        done = False
        total_reward = 0

        states, actions, rewards, log_probs = [], [], [], []
        while not done:
            # Get policy and value from local model
            policy, value = local_model(state)
            action = np.random.choice(len(policy.detach().numpy()), p=policy.detach().numpy())
            log_prob = torch.log(policy[action])

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            total_reward += reward

        # Calculate discounted rewards and advantage
        R = 0 if done else local_model(next_state)[1].item()
        discounted_rewards = []
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        values = torch.cat([local_model(s)[1] for s in states])
        advantages = discounted_rewards - values.detach()

        # Compute losses
        actor_loss = -torch.stack(log_probs) * advantages
        critic_loss = (advantages ** 2).mean()
        total_loss = actor_loss.sum() + critic_loss

        # Update global model
        optimizer.zero_grad()
        total_loss.backward()
        with lock:  # Prevent workers from updating simultaneously
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                global_param.grad = local_param.grad
            optimizer.step()

        # Sync local model with global model
        local_model.load_state_dict(global_model.state_dict())
        print(f"Worker {worker_id}, Episode {episode + 1}, Reward: {total_reward}")

# A3C Training
def train_a3c(env_name, num_workers=4, num_episodes=500, gamma=0.99, lr=0.001):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Global shared model and optimizer
    global_model = ActorCritic(state_dim, action_dim)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)

    # Create worker processes
    lock = multiprocessing.Lock()
    processes = []
    for worker_id in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(worker_id, env_name, global_model, optimizer, num_episodes, gamma, lock))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

# Example usage
if __name__ == "__main__":
    train_a3c(env_name='CartPole-v1')
