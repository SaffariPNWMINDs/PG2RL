import gym
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from pandapower.topology import create_nxgraph
import time

# ===== ACOPF Environment with Graph Representation =====
class ACOPFEnv(gym.Env):
    def __init__(self, network):
        super(ACOPFEnv, self).__init__()
        self.network = network
        self.num_buses = len(network.bus)
        self.num_generators = len(network.gen)
        self.num_loads = len(network.load)
        
        # Observation space: 3 features per bus
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_buses, 3))
        
        # Action space: now includes p_mw, vm_pu, AND q_mvar for each generator
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_generators * 3,))
    
    def _apply_action(self, action):
        for i, gen in enumerate(self.network.gen.index):
            # Active power control
            min_p, max_p = self.network.gen.at[gen, "min_p_mw"], self.network.gen.at[gen, "max_p_mw"]
            self.network.gen.at[gen, "p_mw"] = np.interp(action[i], [-1, 1], [min_p, max_p])
            
            # Voltage control
            self.network.gen.at[gen, "vm_pu"] = np.interp(action[i + self.num_generators], [-1, 1], [0.9, 1.1])
            
            # Reactive power control (added)
            min_q, max_q = self.network.gen.at[gen, "min_q_mvar"], self.network.gen.at[gen, "max_q_mvar"]
            self.network.gen.at[gen, "q_mvar"] = np.interp(action[i + 2*self.num_generators], [-1, 1], [min_q, max_q])
    
    def reset(self):
        for bus_id in self.network.bus.index:
            self.network.bus.at[bus_id, "vm_pu"] = 1.0
        return self._get_graph_representation()
    
    def step(self, action):
        self._apply_action(action)
        try:
            pp.runpp(self.network, algorithm="nr")
            reward = -self._compute_cost()
            done = self._check_constraints()
        except pp.powerflow.LoadflowNotConverged:
            reward = -500
            done = True
        return self._get_graph_representation(), reward, done, {}
    
    def _compute_cost(self):
        cost = sum(0.01 * p**2 + 0.1 * p for p in self.network.gen["p_mw"])
        voltage_deviation = sum(abs(self.network.bus["vm_pu"] - 1.0))
        return cost + 2 * voltage_deviation
    
    def _check_constraints(self):
        voltage_violations = any((self.network.bus["vm_pu"] < 0.9) | (self.network.bus["vm_pu"] > 1.1))
        line_overloads = any(self.network.res_line["loading_percent"] > 110)
        return voltage_violations or line_overloads
    
    def _get_graph_representation(self):
        # Get voltage magnitude for all buses
        vm_pu = self.network.bus["vm_pu"].values

        # Create a zero array for load features with the same length as buses
        p_mw = np.zeros(self.num_buses)
        q_mvar = np.zeros(self.num_buses)

        # Fill the load features at their respective bus indices
        for _, load in self.network.load.iterrows():
            bus_id = int(load["bus"])  # Get bus index
            p_mw[bus_id] = load["p_mw"]
            q_mvar[bus_id] = load["q_mvar"]

        # Stack features correctly
        node_features = np.column_stack((vm_pu, p_mw, q_mvar))

        # Get edge index
        edge_index = torch.tensor(np.array(self.network.line[["from_bus", "to_bus"]].T), dtype=torch.long)

        # Use line impedance (resistance and reactance) as edge features
        edge_attr = torch.tensor(self.network.line[["r_ohm_per_km", "x_ohm_per_km"]].values, dtype=torch.float)

        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
    
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ===== GNN-based TD3 Components =====
class GNNActor(nn.Module):
    def __init__(self, node_features, action_dim):
        super(GNNActor, self).__init__()
        self.conv1 = GATConv(node_features, 128, edge_dim=2)  # edge_dim=2 for r_ohm_per_km and x_ohm_per_km
        self.conv2 = GATConv(128, 128, edge_dim=2)
        self.fc = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index, edge_attr))  # Pass edge_attr
        x = torch.relu(self.conv2(x, edge_index, edge_attr))  # Pass edge_attr
        x = x.mean(dim=0)  # Aggregate node features
        return self.tanh(self.fc(x))

class GNNCritic(nn.Module):
    def __init__(self, node_features, action_dim):
        super(GNNCritic, self).__init__()
        self.conv1 = GATConv(node_features + action_dim, 128, edge_dim=2)  # edge_dim=2
        self.conv2 = GATConv(128, 128, edge_dim=2)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, data, action):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        action = action.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.conv1(x, edge_index, edge_attr))  # Pass edge_attr
        x = torch.relu(self.conv2(x, edge_index, edge_attr))  # Pass edge_attr
        x = x.mean(dim=0)
        return self.fc(x)

class TD3Agent:
    def __init__(self, node_features, action_dim):
        self.actor = GNNActor(node_features, action_dim)
        self.critic1 = GNNCritic(node_features, action_dim)
        self.critic2 = GNNCritic(node_features, action_dim)
        self.target_actor = GNNActor(node_features, action_dim)
        self.target_critic1 = GNNCritic(node_features, action_dim)
        self.target_critic2 = GNNCritic(node_features, action_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=0.001)
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
    
    def select_action(self, state):
        return self.actor(state).detach().cpu().numpy().flatten()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
    
        # Extract components from the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Keep states and next_states as lists of Data objects
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Training step
        with torch.no_grad():
            next_actions = torch.vstack([self.target_actor(state) for state in next_states])
            target_q = torch.min(
                torch.vstack([self.target_critic1(state, action) for state, action in zip(next_states, next_actions)]),
                torch.vstack([self.target_critic2(state, action) for state, action in zip(next_states, next_actions)])
            )
            target_q = rewards + 0.99 * target_q * (1 - dones)

        # Compute critic loss
        critic_loss = 0
        for state, action, tq in zip(states, actions, target_q):
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            critic_loss += (q1 - tq).pow(2).mean() + (q2 - tq).pow(2).mean()
    
        critic_loss /= self.batch_size

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        actor_loss = -torch.mean(torch.vstack([self.critic1(state, self.actor(state)) for state in states]))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

def calculate_mape_components(predicted, actual, num_gen):
    """Calculate MAPE separately for p_mw, vm_pu, and q_mvar"""
    epsilon = 1e-10
    
    # Split the action vector into components
    pred_p_mw = predicted[:num_gen]
    pred_vm_pu = predicted[num_gen:2*num_gen]
    pred_q_mvar = predicted[2*num_gen:]
    
    actual_p_mw = actual[:num_gen]
    actual_vm_pu = actual[num_gen:2*num_gen]
    actual_q_mvar = actual[2*num_gen:]
    
    # Calculate MAPE for each component
    actual_p_mw = np.where(np.abs(actual_p_mw) < epsilon, epsilon, actual_p_mw)
    mape_p = np.mean(np.abs((actual_p_mw - pred_p_mw) / actual_p_mw))
    
    actual_vm_pu = np.where(np.abs(actual_vm_pu) < epsilon, epsilon, actual_vm_pu)
    mape_v = np.mean(np.abs((actual_vm_pu - pred_vm_pu) / actual_vm_pu))
    
    actual_q_mvar = np.where(np.abs(actual_q_mvar) < epsilon, epsilon, actual_q_mvar)
    mape_q = np.mean(np.abs((actual_q_mvar - pred_q_mvar) / actual_q_mvar))
    
    return mape_p, mape_v, mape_q

def calculate_rmse_components(predicted, actual, num_gen):
    """Calculate RMSE separately for p_mw, vm_pu, and q_mvar"""
    
    # Split the action vector into components
    pred_p_mw = predicted[:num_gen]
    pred_vm_pu = predicted[num_gen:2*num_gen]
    pred_q_mvar = predicted[2*num_gen:]
    
    actual_p_mw = actual[:num_gen]
    actual_vm_pu = actual[num_gen:2*num_gen]
    actual_q_mvar = actual[2*num_gen:]
    
    # Calculate RMSE for each component
    rmse_p = np.sqrt(np.mean((actual_p_mw - pred_p_mw) ** 2))
    rmse_v = np.sqrt(np.mean((actual_vm_pu - pred_vm_pu) ** 2))
    rmse_q = np.sqrt(np.mean((actual_q_mvar - pred_q_mvar) ** 2))
    
    return rmse_p, rmse_v, rmse_q

def train_td3(env, agent, episodes=100, update_after=10, update_every=5):
    rewards = []
    mape_p_values = []  # For p_mw
    mape_v_values = []  # For vm_pu
    mape_q_values = []  # For q_mvar
    rmse_p_values = []
    rmse_v_values = []
    rmse_q_values = []


    num_gen = env.network.gen.shape[0]
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_mape_p = 0
        episode_mape_v = 0
        episode_mape_q = 0
        episode_steps = 0
        episode_rmse_p = 0
        episode_rmse_v = 0
        episode_rmse_q = 0

        while not done:
            action = agent.select_action(state)
            reference_action = get_reference_action(env.network)
            
            # Calculate MAPE for each component
            mape_p, mape_v, mape_q = calculate_mape_components(action, reference_action, num_gen)
            episode_mape_p += mape_p
            episode_mape_v += mape_v
            episode_mape_q += mape_q
            rmse_p, rmse_v, rmse_q = calculate_rmse_components(action, reference_action, num_gen)
            episode_rmse_p += rmse_p
            episode_rmse_v += rmse_v
            episode_rmse_q += rmse_q

            episode_steps += 1

            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(agent.replay_buffer) > update_after:
                for _ in range(update_every):
                    agent.train()

        # Calculate average MAPE for each component
        avg_mape_p = episode_mape_p / episode_steps
        avg_mape_v = episode_mape_v / episode_steps
        avg_mape_q = episode_mape_q / episode_steps
        
        mape_p_values.append(avg_mape_p)
        mape_v_values.append(avg_mape_v)
        mape_q_values.append(avg_mape_q)

        avg_rmse_p = episode_rmse_p / episode_steps
        avg_rmse_v = episode_rmse_v / episode_steps
        avg_rmse_q = episode_rmse_q / episode_steps

        rmse_p_values.append(avg_rmse_p)
        rmse_v_values.append(avg_rmse_v)
        rmse_q_values.append(avg_rmse_q)

        
        rewards.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
        print(f"  MAPE - P: {avg_mape_p:.4f}, V: {avg_mape_v:.4f}, Q: {avg_mape_q:.4f}")
        print(f"  RMSE - P: {avg_rmse_p:.4f}, V: {avg_rmse_v:.4f}, Q: {avg_rmse_q:.4f}")

    training_time = time.time() - start_time
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.scatter(range(len(rewards)), rewards, color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    
    plt.subplot(1, 3, 2)
    plt.plot(mape_p_values, label="P (MW)")
    plt.plot(mape_v_values, label="V (pu)")
    plt.plot(mape_q_values, label="Q (MVAR)")
    plt.xlabel("Episodes")
    plt.ylabel("MAPE (%)")
    plt.title("Component-wise MAPE")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(mape_p_values, label="P", color='blue')
    plt.plot(mape_v_values, label="V", color='orange')
    plt.plot(mape_q_values, label="Q", color='green')
    plt.yscale('log')
    plt.xlabel("Episodes")
    plt.ylabel("MAPE (%) - Log Scale")
    plt.title("Log-Scale MAPE Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return training_time

def test_td3(env, agent, test_episodes=10):
    test_rewards = []
    test_mape_p = []
    test_mape_v = []
    test_mape_q = []
    
    num_gen = env.network.gen.shape[0]
    start_time = time.time()

    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_mape_p = 0
        episode_mape_v = 0
        episode_mape_q = 0
        episode_steps = 0

        while not done:
            action = agent.select_action(state)
            reference_action = get_reference_action(env.network)
            
            mape_p, mape_v, mape_q = calculate_mape_components(action, reference_action, num_gen)
            episode_mape_p += mape_p
            episode_mape_v += mape_v
            episode_mape_q += mape_q
            episode_steps += 1
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        avg_mape_p = episode_mape_p / episode_steps
        avg_mape_v = episode_mape_v / episode_steps
        avg_mape_q = episode_mape_q / episode_steps
        
        test_rewards.append(total_reward)
        test_mape_p.append(avg_mape_p)
        test_mape_v.append(avg_mape_v)
        test_mape_q.append(avg_mape_q)
        
        print(f"Test Episode {episode+1}, Total Reward: {total_reward}")
        print(f"  MAPE - P: {avg_mape_p:.4f}%, V: {avg_mape_v:.4f}%, Q: {avg_mape_q:.4f}%")

    testing_time = time.time() - start_time
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(test_rewards)
    plt.scatter(range(len(test_rewards)), test_rewards, color='r')
    plt.xlabel("Test Episodes")
    plt.ylabel("Total Reward")
    plt.title("Testing Rewards")
    
    plt.subplot(1, 3, 2)
    plt.plot(test_mape_p, label="P (MW)", color='blue')
    plt.plot(test_mape_v, label="V (pu)", color='orange')
    plt.plot(test_mape_q, label="Q (MVAR)", color='green')
    plt.xlabel("Test Episodes")
    plt.ylabel("MAPE (%)")
    plt.title("Component-wise MAPE")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(test_mape_p, label="P", color='blue')
    plt.plot(test_mape_v, label="V", color='orange')
    plt.plot(test_mape_q, label="Q", color='green')
    plt.yscale('log')
    plt.xlabel("Test Episodes")
    plt.ylabel("MAPE (%) - Log Scale")
    plt.title("Log-Scale MAPE Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return testing_time

def get_reference_action(network):
    try:
        # Try to solve OPF
        pp.runopp(network)  # Run optimal power flow
        reference_p_mw = network.res_gen["p_mw"].values  # Optimal active power
        reference_vm_pu = network.res_gen["vm_pu"].values  # Optimal voltage setpoints
        reference_q_mvar = network.res_gen["q_mvar"].values  # Optimal reactive power
    except:
        # Fall back to power flow if OPF fails
        pp.runpp(network)  # Run standard power flow
        reference_p_mw = network.res_gen["p_mw"].values
        reference_vm_pu = network.res_gen["vm_pu"].values
        reference_q_mvar = network.res_gen["q_mvar"].values

    # Combine into a single action vector
    reference_action = np.concatenate([reference_p_mw, reference_vm_pu, reference_q_mvar])

    # Scale reference action to [-1, 1]
    min_p_mw = network.gen["min_p_mw"].values
    max_p_mw = network.gen["max_p_mw"].values
    min_q_mvar = network.gen["min_q_mvar"].values
    max_q_mvar = network.gen["max_q_mvar"].values
    min_vm_pu = 0.9
    max_vm_pu = 1.1
    
    scaled_action = scale_reference_action(
        reference_action, 
        min_p_mw, max_p_mw,
        min_q_mvar, max_q_mvar,
        min_vm_pu, max_vm_pu
    )
    
    return scaled_action

def scale_reference_action(reference_action, min_p_mw, max_p_mw, min_q_mvar, max_q_mvar, min_vm_pu=0.9, max_vm_pu=1.1):
    num_gen = len(min_p_mw)
    
    # Split reference_action into p_mw, vm_pu, and q_mvar
    p_mw = reference_action[:num_gen]  # First num_gen values are p_mw
    vm_pu = reference_action[num_gen:2*num_gen]  # Next num_gen values are vm_pu
    q_mvar = reference_action[2*num_gen:]  # Last num_gen values are q_mvar

    # Scale p_mw to [-1, 1]
    scaled_p_mw = 2 * (p_mw - min_p_mw) / (max_p_mw - min_p_mw) - 1

    # Scale vm_pu to [-1, 1]
    scaled_vm_pu = 2 * (vm_pu - min_vm_pu) / (max_vm_pu - min_vm_pu) - 1

    # Scale q_mvar to [-1, 1]
    scaled_q_mvar = 2 * (q_mvar - min_q_mvar) / (max_q_mvar - min_q_mvar) - 1

    # Combine scaled values
    scaled_action = np.concatenate([scaled_p_mw, scaled_vm_pu, scaled_q_mvar])
    return scaled_action

def plot_graph(network):
    import matplotlib.pyplot as plt
    import networkx as nx
    from pandapower.topology import create_nxgraph

    pp.runpp(network)
    G = create_nxgraph(network)
    pos = nx.spring_layout(G, seed=42)

    # Labels for buses
    node_labels = {
        bus: f"Bus {bus}\nV={network.res_bus.at[bus, 'vm_pu']:.2f} pu\n"
             f"P={network.load.loc[network.load['bus'] == bus, 'p_mw'].sum():.1f} MW\n"
             f"Q={network.load.loc[network.load['bus'] == bus, 'q_mvar'].sum():.1f} MVAR"
        for bus in network.bus.index
    }

    # Labels for edges (R and X)
    edge_labels = {
        (row["from_bus"], row["to_bus"]): f"R={row['r_ohm_per_km']:.2f}Ω\nX={row['x_ohm_per_km']:.2f}Ω"
        for _, row in network.line.iterrows()
    }

    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#fcca46', node_size=1200,
                           edgecolors='black', linewidths=1.5)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='#386641', width=2)

    # Draw simple bus labels
    nx.draw_networkx_labels(G, pos, labels={bus: str(bus) for bus in network.bus.index},
                            font_size=10, font_weight='bold')

    # Draw extended node labels slightly above each node
    offset_pos = {bus: (x, y + 0.12) for bus, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, offset_pos, labels=node_labels,
                            font_size=8, font_color='darkblue')

    # Draw edge impedance labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, font_color='darkred')

    # plt.title("Graph Representation of IEEE 9-Bus System", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_with_gaussian_noise(env, agent, noise_levels, test_episodes=5):
    """Tests PG2RL model with Gaussian noise added to node features at inference time."""
    results = []
    num_gen = env.network.gen.shape[0]

    for sigma in noise_levels:
        mape_p_total, mape_v_total, mape_q_total = 0, 0, 0
        steps_total = 0

        for _ in range(test_episodes):
            state = env.reset()
            done = False

            while not done:
                # Add Gaussian noise only to the actor input
                noisy_input = state.clone()
                noisy_input.x = state.x + torch.normal(mean=0.0, std=sigma, size=state.x.size())

                action = agent.select_action(noisy_input)
                reference_action = get_reference_action(env.network)

                mape_p, mape_v, mape_q = calculate_mape_components(action, reference_action, num_gen)
                mape_p_total += mape_p
                mape_v_total += mape_v
                mape_q_total += mape_q
                steps_total += 1

                next_state, _, done, _ = env.step(action)
                state = next_state  # Continue simulation with original env state

        avg_mape_p = mape_p_total / steps_total
        avg_mape_v = mape_v_total / steps_total
        avg_mape_q = mape_q_total / steps_total

        results.append({
            "Noise Std. Dev. (σ)": sigma,
            "Pg MAPE (%)": avg_mape_p * 100,
            "Qg MAPE (%)": avg_mape_q * 100,
            "Vi MAPE (%)": avg_mape_v * 100
        })

    df_results = pd.DataFrame(results)
    return df_results

# ===== Main Execution =====
if __name__ == "__main__":
    # Load the IEEE 9-bus system
    network = pn.case9()
    plot_graph(network)
    # env = ACOPFEnv(network)
    # agent = TD3Agent(node_features=3, action_dim=network.gen.shape[0] * 3)

    # # Training
    # training_time = train_td3(env, agent, episodes=100)
    # print(f"Training completed in {training_time:.2f} seconds")

    # # # Testing
    # # testing_time = test_td3(env, agent, test_episodes=10)
    # # print(f"Testing completed in {testing_time:.2f} seconds")

    # # Robustness testing with noise
    # df_noise_results = test_with_gaussian_noise(env, agent, noise_levels=[0.0, 0.1, 1, 10], test_episodes=10)
    # # plot_noise_impact(df_noise_results)

    # # Optional: print as a table
    # print("\nEffect of Gaussian Noise on MAPE:")
    # print(df_noise_results.to_string(index=False))

    # Optional: Plot the network graph
    # plot_graph(network)