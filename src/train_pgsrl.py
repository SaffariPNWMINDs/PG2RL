"""
Reference implementation accompanying:

    Physics-Guided Graph Safe Reinforcement Learning for High-Fidelity and
    Scalable Alternating Current Optimal Power Flow
    Y. P. Singh, M. Saffari and A. Asrari, Processes, 2026.

Additional materials are available from the corresponding author
(msaffari@pnw.edu) upon reasonable request.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from pandapower.topology import create_nxgraph
import time
from thop import profile
import copy
import pandas as pd

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

# ===== Graph Mamba Layer (using MessagePassing like GAT) =====
class GraphMambaConv(MessagePassing):
    """
    Graph Mamba layer - drop-in replacement for GATConv
    Uses message passing with gating mechanism
    """
    def __init__(self, in_channels, out_channels, edge_dim=2):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main transformation
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Gate for selective propagation (Mamba-style)
        self.gate = nn.Linear(in_channels, out_channels)
        
        # Edge embedding (same as GAT)
        self.edge_lin = nn.Linear(edge_dim, out_channels)
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.zeros_(self.edge_lin.bias)
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Pad edge_attr with zeros for self-loops
        num_self_loops = x.size(0)
        self_loop_attr = torch.zeros(num_self_loops, edge_attr.size(1), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Transform input first
        x_transformed = self.lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_transformed, norm=norm, edge_attr=edge_attr)
        
        # Apply gating (Mamba-style selective mechanism)
        gate_val = torch.sigmoid(self.gate(x))
        out = out * gate_val
        
        return F.relu(out)
    
    def message(self, x_j, norm, edge_attr):
        # Add edge embedding to transformed node features
        edge_emb = self.edge_lin(edge_attr)
        msg = x_j + edge_emb
        return norm.view(-1, 1) * msg

# ===== GNN-based TD3 Components (with Graph Mamba) =====
class GNNActor(nn.Module):
    def __init__(self, node_features, action_dim):
        super(GNNActor, self).__init__()
        self.conv1 = GraphMambaConv(node_features, 128, edge_dim=2)
        self.conv2 = GraphMambaConv(128, 128, edge_dim=2)
        self.fc = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = x.mean(dim=0)  # Aggregate node features
        return self.tanh(self.fc(x))

class GNNCritic(nn.Module):
    def __init__(self, node_features, action_dim):
        super(GNNCritic, self).__init__()
        self.conv1 = GraphMambaConv(node_features + action_dim, 128, edge_dim=2)
        self.conv2 = GraphMambaConv(128, 128, edge_dim=2)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, data, action):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        action = action.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
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
        
        wd = 1e-4  # regularization strength (try 1e-5 if you want it even milder)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001, weight_decay=wd)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=0.001,
            weight_decay=wd
        )
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
    
    def select_action_with_noise(self, state, noise_std=0.2):
        action = self.actor(state).detach().cpu().numpy().flatten()
        noise = np.random.normal(0, noise_std, size=action.shape)
        action += noise
        return np.clip(action, -1, 1)  # Keep action within bounds

    
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

def train_td3(env, agent, episodes=100, update_after=10, update_every=5):
    rewards = []
    mape_p_list, mape_q_list, mape_v_list = [], [], []

    min_p = env.network.gen["min_p_mw"].values
    max_p = env.network.gen["max_p_mw"].values
    min_q = env.network.gen["min_q_mvar"].values
    max_q = env.network.gen["max_q_mvar"].values

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        ep_mape_p, ep_mape_q, ep_mape_v = [], [], []

        while not done:
            # actor action in [-1,1]
            action_scaled = agent.select_action_with_noise(state, noise_std=0.1)

            # reference OPF action
            ref_scaled, (p_ref, v_ref, q_ref) = get_reference_action_pvq(env.network)

            # unscale predicted action to physical for MAPE
            p_pred, v_pred, q_pred = unscale_action_pvq(action_scaled, min_p, max_p, min_q, max_q)

            # MAPE in physical domain
            ep_mape_p.append(mape(p_pred, p_ref))
            ep_mape_q.append(mape(q_pred, q_ref))
            ep_mape_v.append(mape(v_pred, v_ref))

            next_state, reward, done, _ = env.step(action_scaled)
            agent.replay_buffer.append((state, action_scaled, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(agent.replay_buffer) > update_after:
                for _ in range(update_every):
                    agent.train()

        rewards.append(total_reward)

        # avg episode MAPEs
        avg_mp = float(np.mean(ep_mape_p)) if ep_mape_p else np.nan
        avg_mq = float(np.mean(ep_mape_q)) if ep_mape_q else np.nan
        avg_mv = float(np.mean(ep_mape_v)) if ep_mape_v else np.nan

        mape_p_list.append(avg_mp)
        mape_q_list.append(avg_mq)
        mape_v_list.append(avg_mv)

        print(f"Episode {episode+1}, Reward={total_reward:.2f}, "
              f"MAPE(P)={avg_mp:.2f}%, MAPE(Q)={avg_mq:.2f}%, MAPE(V)={avg_mv:.2f}%")

    # Optional: plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(rewards); plt.title("Reward"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.subplot(1,2,2); 
    plt.plot(mape_p_list, label="P"); plt.plot(mape_q_list, label="Q"); plt.plot(mape_v_list, label="V")
    plt.title("MAPE"); plt.xlabel("Episode"); plt.ylabel("%"); plt.legend()
    plt.tight_layout(); plt.show()

    return rewards, mape_p_list, mape_q_list, mape_v_list


def plot_rewards(rewards, window_size=10):
    """
    Plot rewards with scientific appearance - MAXIMUM VISIBILITY.
    
    Parameters:
    - rewards: List or array of episode rewards  
    - window_size: Window size for moving average smoothing
    """
    from scipy.ndimage import uniform_filter1d
    
    episodes = np.arange(len(rewards))
    rewards_array = np.array(rewards)
    
    # Calculate smoothed rewards using moving average
    smoothed_rewards = uniform_filter1d(rewards_array, size=window_size, mode='nearest')
    
    # Create figure (larger size)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw rewards (MAXIMUM VISIBILITY)
    ax.plot(episodes, rewards_array, color='#3498DB', alpha=0.75, linewidth=1.2, zorder=1)
    
    # Plot smoothed rewards (dark blue, thick)
    ax.plot(episodes, smoothed_rewards, color='#1A5490', linewidth=3.5, zorder=2)
    
    # Labels (LARGE and bold)
    ax.set_xlabel('Episode', fontsize=21, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=21, fontweight='bold')
    
    # Grid
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.15, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Text annotation (LARGER font)
    ax.text(0.98, 0.02, 'Accumulate Reward in Each Episode', 
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, 
                     edgecolor='gray', linewidth=0.8))
    
    # Tick parameters (LARGE and bold)
    ax.tick_params(axis='both', labelsize=19)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Borders
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#666666')
    
    # Background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig("reward_per_episode_mamba.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig("reward_per_episode_mamba.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nReward plot saved to reward_per_episode_mamba.png and .pdf")
    
    plt.show()

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

def measure_flops(actor, sample_state):
    # sample_state is a PyG Data object
    flops, params = profile(
        actor,
        inputs=(sample_state,),
        verbose=False
    )
    return flops, params

def measure_inference_time(agent, env, runs=300):
    state = env.reset()

    # Warm-up (important for fair timing)
    for _ in range(30):
        _ = agent.actor(state)

    times = []
    for _ in range(runs):
        start = time.time()
        _ = agent.actor(state)
        end = time.time()
        times.append(end - start)

    return np.mean(times), np.std(times)

def evaluate_mape_after_training(env, agent, n_tests=50):
    min_p = env.network.gen["min_p_mw"].values
    max_p = env.network.gen["max_p_mw"].values
    min_q = env.network.gen["min_q_mvar"].values
    max_q = env.network.gen["max_q_mvar"].values

    Pp, Pr, Qp, Qr, Vp, Vr = [], [], [], [], [], []

    for _ in range(n_tests):
        state = env.reset()
        with torch.no_grad():
            a = agent.actor(state).cpu().numpy().flatten()  # [-1,1]

        _, (p_ref, v_ref, q_ref) = get_reference_action_pvq(env.network)
        p_pred, v_pred, q_pred = unscale_action_pvq(a, min_p, max_p, min_q, max_q)

        Pp.append(p_pred); Pr.append(p_ref)
        Qp.append(q_pred); Qr.append(q_ref)
        Vp.append(v_pred); Vr.append(v_ref)

    Pp = np.concatenate(Pp); Pr = np.concatenate(Pr)
    Qp = np.concatenate(Qp); Qr = np.concatenate(Qr)
    Vp = np.concatenate(Vp); Vr = np.concatenate(Vr)

    print("\n==== Final Evaluation (MAPE) ====")
    print(f"MAPE(Pg): {mape(Pp, Pr):.2f}%")
    print(f"MAPE(Qg): {mape(Qp, Qr):.2f}%")
    print(f"MAPE(V):  {mape(Vp, Vr):.2f}%")

def compute_violation_metrics(net, vmin=0.9, vmax=1.1, smax_pct=100.0):
    """Return feasibility flags + magnitudes after a successful runpp."""
    vm = net.res_bus["vm_pu"].values
    vdev = np.sum(np.maximum(0, vm - vmax) + np.maximum(0, vmin - vm))

    loading = net.res_line["loading_percent"].values
    sover = np.sum(np.maximum(0, loading - smax_pct))

    voltage_violation = np.any((vm < vmin) | (vm > vmax))
    line_overload = np.any(loading > smax_pct)

    return voltage_violation, line_overload, vdev, sover

def scale_action_pvq(p_mw, vm_pu, q_mvar, min_p, max_p, min_q, max_q, min_vm=0.9, max_vm=1.1):
    """Scale physical (P, V, Q) to [-1,1]"""
    p_s = 2 * (p_mw - min_p) / (max_p - min_p + 1e-9) - 1
    v_s = 2 * (vm_pu - min_vm) / (max_vm - min_vm + 1e-9) - 1
    q_s = 2 * (q_mvar - min_q) / (max_q - min_q + 1e-9) - 1
    return np.concatenate([p_s, v_s, q_s])

def unscale_action_pvq(action_scaled, min_p, max_p, min_q, max_q, min_vm=0.9, max_vm=1.1):
    """Unscale [-1,1] action to physical (P, V, Q)"""
    ng = len(min_p)
    p_s = action_scaled[:ng]
    v_s = action_scaled[ng:2*ng]
    q_s = action_scaled[2*ng:3*ng]

    p = 0.5 * (p_s + 1) * (max_p - min_p) + min_p
    v = 0.5 * (v_s + 1) * (max_vm - min_vm) + min_vm
    q = 0.5 * (q_s + 1) * (max_q - min_q) + min_q
    return p, v, q

def mape(y_pred, y_true, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def get_reference_action_pvq(network):
    try:
        pp.runopp(network)
        p_ref = network.res_gen["p_mw"].values
        v_ref = network.res_gen["vm_pu"].values
        q_ref = network.res_gen["q_mvar"].values
    except:
        pp.runpp(network)
        p_ref = network.res_gen["p_mw"].values
        v_ref = network.res_gen["vm_pu"].values
        # If q not available from pp in your setup, fall back to current gen q
        q_ref = network.res_gen["q_mvar"].values if "q_mvar" in network.res_gen else network.gen["q_mvar"].values

    min_p = network.gen["min_p_mw"].values
    max_p = network.gen["max_p_mw"].values
    min_q = network.gen["min_q_mvar"].values
    max_q = network.gen["max_q_mvar"].values

    ref_scaled = scale_action_pvq(p_ref, v_ref, q_ref, min_p, max_p, min_q, max_q)
    return ref_scaled, (p_ref, v_ref, q_ref)


def n_1_contingency_test(base_network, agent, vmin=0.9, vmax=1.1, smax_pct=100.0,
                        algorithm="nr", verbose=True):
    """
    N-1 test: remove each line one at a time, apply actor action, run power flow,
    and record convergence + violations.
    """
    # We will NOT modify the original network permanently
    net0 = copy.deepcopy(base_network)

    n_lines = len(net0.line)
    results = []

    for outage_idx in net0.line.index:
        net = copy.deepcopy(net0)

        # Drop this line (N-1 outage)
        net.line.drop(outage_idx, inplace=True)

        # Create env for this contingency network (so state matches topology)
        env = ACOPFEnv(net)

        # Reset to get graph state
        state = env.reset()

        # Actor gives action in [-1,1]
        action = agent.actor(state).detach().cpu().numpy().flatten()

        # Apply action to gens
        env._apply_action(action)

        # Run PF and score
        converged = True
        try:
            pp.runpp(net, algorithm=algorithm)
            voltage_violation, line_overload, vdev, sover = compute_violation_metrics(
                net, vmin=vmin, vmax=vmax, smax_pct=smax_pct
            )
        except pp.powerflow.LoadflowNotConverged:
            converged = False
            voltage_violation, line_overload, vdev, sover = True, True, np.nan, np.nan

        results.append({
            "outage_line": int(outage_idx),
            "converged": converged,
            "voltage_violation": bool(voltage_violation) if converged else True,
            "line_overload": bool(line_overload) if converged else True,
            "Vdev": float(vdev) if converged else np.nan,
            "Soverflow": float(sover) if converged else np.nan
        })

        if verbose:
            print(f"[N-1] line {outage_idx}: converged={converged}, "
                  f"Vviol={voltage_violation}, Lover={line_overload}, "
                  f"Vdev={vdev}, Sover={sover}")

    # Aggregate
    df = pd.DataFrame(results)
    total = len(df)
    conv_rate = 100.0 * df["converged"].mean()

    # Violation rates computed only among converged cases
    df_conv = df[df["converged"] == True]
    if len(df_conv) > 0:
        vviol_rate = 100.0 * df_conv["voltage_violation"].mean()
        lover_rate = 100.0 * df_conv["line_overload"].mean()
        avg_vdev = df_conv["Vdev"].mean()
        avg_sover = df_conv["Soverflow"].mean()
    else:
        vviol_rate, lover_rate, avg_vdev, avg_sover = 100.0, 100.0, np.nan, np.nan

    summary = {
        "N-1 cases": total,
        "PF convergence rate (%)": conv_rate,
        "Voltage violation rate (%)": vviol_rate,
        "Line overload rate (%)": lover_rate,
        "Avg Vdev": avg_vdev,
        "Avg Soverflow": avg_sover
    }

    return df, summary

# ===== Main Execution =====
if __name__ == "__main__":
    print("="*70)
    print("GRAPH MAMBA TRAINING FOR ACOPF")
    print("="*70)
    
    # Load the IEEE 9-bus system
    network = pn.case300()
    env = ACOPFEnv(network)
    agent = TD3Agent(node_features=3, action_dim=network.gen.shape[0] * 3)

    sample_state = env.reset()
    flops, params = measure_flops(agent.actor, sample_state)
    print(f"Actor FLOPs: {flops/1e6:.2f} MFLOPs")
    print(f"Actor Params: {params/1e6:.2f} M")

    mean_t, std_t = measure_inference_time(agent, env)
    print(f"Inference time per step: {mean_t*1000:.3f} ± {std_t*1000:.3f} ms")

    train_td3(env, agent, episodes=100)
    evaluate_mape_after_training(env, agent, n_tests=50)
    df_n1, summary_n1 = n_1_contingency_test(network, agent, smax_pct=100.0, verbose=False)

    print("\nN-1 Summary:")
    for k, v in summary_n1.items():
        print(f"{k}: {v}")

    print("\nFirst 10 N-1 cases:")
    print(df_n1.head(10))
    
    print("\n"+"="*70)
    print("Training complete!")
    print("="*70)
