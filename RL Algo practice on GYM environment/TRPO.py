import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

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

def conjugate_gradient(Ax, b, max_iterations=10, tolerance=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)

    for _ in range(max_iterations):
        Ap = Ax(p)
        alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_r_dot_r = torch.dot(r, r)

        if new_r_dot_r < tolerance:
            break

        beta = new_r_dot_r / r_dot_r
        p = r + beta * p
        r_dot_r = new_r_dot_r

    return x

def fisher_vector_product(actor, states, p, damping=1e-2):
    p.detach()
    action_probs = actor(states)
    action_probs = torch.clamp(action_probs, 1e-8, 1 - 1e-8)
    kl = (action_probs * torch.log(action_probs / action_probs.detach())).sum(dim=1).mean()

    grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    grad_vector_product = torch.dot(flat_grads, p)

    hvp = torch.autograd.grad(grad_vector_product, actor.parameters())
    hvp = torch.cat([g.contiguous().view(-1) for g in hvp])

    return hvp + damping * p

def update_parameters(parameters, step):
    index = 0
    for param in parameters:
        size = param.numel()
        param.data += step[index:index + size].view(param.size())
        index += size

def train_trpo(env, actor, critic, num_episodes=1000, gamma=0.99, delta=0.01, max_steps=1000):
    total_rewards_per_episode = []
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)

        states, actions, rewards, log_probs = [], [], [], []
        total_reward = 0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            step_count += 1
            action_probs = actor(state)
            action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
            log_prob = torch.log(action_probs[action])

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            total_reward += reward

        total_rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # Convert collected data to tensors
        states = torch.stack(states).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        log_probs = torch.stack(log_probs)

        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns).float()

        values = critic(states).squeeze(1)
        advantages = (returns - values).detach()

        # Update critic
        critic_optimizer.zero_grad()
        value_loss = nn.MSELoss()(values, returns)
        value_loss.backward()
        critic_optimizer.step()

        # Compute policy gradient
        new_action_probs = actor(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_ratio = torch.exp(torch.log(new_action_probs) - log_probs.detach())
        surrogate_loss = (policy_ratio * advantages).mean()

        grads = torch.autograd.grad(surrogate_loss, actor.parameters())
        flat_grads = torch.cat([g.view(-1) for g in grads])

        # Compute step direction using conjugate gradient
        Ax = lambda p: fisher_vector_product(actor, states, p)
        step_direction = conjugate_gradient(Ax, flat_grads)

        # Compute step size
        step_size = torch.sqrt(2 * delta / torch.dot(step_direction, Ax(step_direction)))
        step = step_size * step_direction

        # Update actor parameters
        update_parameters(actor.parameters(), step)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress (TRPO)")
    plt.legend()
    plt.show()

# Main execution
env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

train_trpo(env, actor, critic, num_episodes=1000, max_steps=1000)