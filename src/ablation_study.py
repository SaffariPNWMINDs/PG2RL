"""
Reference implementation accompanying:

    Physics-Guided Graph Safe Reinforcement Learning for High-Fidelity and
    Scalable Alternating Current Optimal Power Flow
    Y. P. Singh, M. Saffari and A. Asrari, Processes, 2026.

Additional materials are available from the corresponding author
(msaffari@pnw.edu) upon reasonable request.
"""

"""
PGSRL Ablation Study for IEEE Access Paper Revision

This script implements the ablation study requested by reviewers:
(a) Full PGSRL - Complete model with all components
(b) GCN variant - Replace Graph Mamba with standard GCN
(c) No Gating - Remove gating mechanism from Graph Mamba
(d) No Physics-Guided Reward - Use cost-only reward (no constraint penalties)
(e) Single Critic - Replace TD3 twin critics with single critic (DDPG-style)

Test Systems: IEEE 30-bus and IEEE 118-bus
Metrics: MAPE (Pg, Qg, V), Constraint Violation Rate, Training Convergence
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandapower as pp
import pandapower.networks as pn
from gym import spaces
from collections import deque
import random
import time
import warnings
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
import copy

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# ENVIRONMENT CLASSES
# ============================================================================

class ACOPFEnv(gym.Env):
    """Base ACOPF Environment with physics-guided reward"""
    def __init__(self, network, lambda1=1.0, lambda2=1.0, use_physics_reward=True):
        super(ACOPFEnv, self).__init__()
        self.network = network
        self.num_buses = len(network.bus)
        self.num_generators = len(network.gen)
        self.num_loads = len(network.load)
        self.lambda1 = lambda1  # Voltage penalty weight
        self.lambda2 = lambda2  # Line overload penalty weight
        self.use_physics_reward = use_physics_reward
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_buses, 3))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_generators * 3,))
        
        # Store ground truth from OPF solver
        self.ground_truth = None
        self._compute_ground_truth()
        
        # Track constraint violations
        self.voltage_violations = 0
        self.line_violations = 0
        self.total_steps = 0
    
    def _compute_ground_truth(self):
        """Compute ground truth using Pandapower OPF"""
        try:
            pp.runopp(self.network)
            self.ground_truth = {
                'Pg': self.network.res_gen['p_mw'].values.copy(),
                'Qg': self.network.res_gen['q_mvar'].values.copy(),
                'V': self.network.res_bus['vm_pu'].values.copy()
            }
        except:
            self.ground_truth = None
    
    def reset(self):
        for bus_id in self.network.bus.index:
            self.network.bus.at[bus_id, "vm_pu"] = 1.0
        return self._get_graph_representation()
    
    def step(self, action):
        self._apply_action(action)
        self.total_steps += 1
        try:
            pp.runpp(self.network, algorithm="nr")
            reward = self._compute_reward()
            done = self._check_constraints()
            
            # Track violations
            if self._has_voltage_violation():
                self.voltage_violations += 1
            if self._has_line_violation():
                self.line_violations += 1
                
        except pp.powerflow.LoadflowNotConverged:
            reward = -500
            done = True
        return self._get_graph_representation(), reward, done, {}
    
    def _apply_action(self, action):
        for i, gen in enumerate(self.network.gen.index):
            # Active power
            min_p = self.network.gen.at[gen, "min_p_mw"]
            max_p = self.network.gen.at[gen, "max_p_mw"]
            self.network.gen.at[gen, "p_mw"] = np.interp(action[i], [-1, 1], [min_p, max_p])
            
            # Voltage magnitude
            self.network.gen.at[gen, "vm_pu"] = np.interp(
                action[i + self.num_generators], [-1, 1], [0.9, 1.1]
            )
            
            # Reactive power (if action space includes it)
            if len(action) > 2 * self.num_generators:
                min_q = self.network.gen.at[gen, "min_q_mvar"] if "min_q_mvar" in self.network.gen.columns else -100
                max_q = self.network.gen.at[gen, "max_q_mvar"] if "max_q_mvar" in self.network.gen.columns else 100
                # Note: Q is typically determined by power flow, but we can set target
    
    def _compute_reward(self):
        # Generation cost (Eq. 16 in paper)
        cost = sum(0.01 * p**2 + 0.1 * p for p in self.network.gen["p_mw"])
        R_t = -cost
        
        if not self.use_physics_reward:
            return R_t
        
        # Voltage deviation penalty (Eq. 17)
        V_dev = 0
        for i, vm in enumerate(self.network.res_bus["vm_pu"]):
            V_max = 1.1  # Upper voltage limit
            V_min = 0.9  # Lower voltage limit
            V_dev += max(0, vm - V_max) + max(0, V_min - vm)
        
        # Line overload penalty (Eq. 18)
        S_overflow = 0
        for loading in self.network.res_line["loading_percent"]:
            S_max = 100  # 100% loading limit
            S_overflow += max(0, loading - S_max)
        
        # Physics-guided reward (Eq. 19)
        r_t = R_t - self.lambda1 * V_dev - self.lambda2 * S_overflow
        return r_t
    
    def _has_voltage_violation(self):
        return any((self.network.res_bus["vm_pu"] < 0.9) | (self.network.res_bus["vm_pu"] > 1.1))
    
    def _has_line_violation(self):
        return any(self.network.res_line["loading_percent"] > 100)
    
    def _check_constraints(self):
        return self._has_voltage_violation() or self._has_line_violation()
    
    def _get_graph_representation(self):
        vm_pu = self.network.bus["vm_pu"].values
        p_mw = np.zeros(self.num_buses)
        q_mvar = np.zeros(self.num_buses)
        
        for _, load in self.network.load.iterrows():
            bus_id = int(load["bus"])
            if bus_id < self.num_buses:
                p_mw[bus_id] = load["p_mw"]
                q_mvar[bus_id] = load["q_mvar"]
        
        node_features = np.column_stack((vm_pu, p_mw, q_mvar))
        edge_index = torch.tensor(
            np.array(self.network.line[["from_bus", "to_bus"]].T), dtype=torch.long
        )
        edge_attr = torch.tensor(
            self.network.line[["r_ohm_per_km", "x_ohm_per_km"]].values, dtype=torch.float
        )
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_constraint_violation_rate(self):
        if self.total_steps == 0:
            return 0.0, 0.0
        return (self.voltage_violations / self.total_steps * 100,
                self.line_violations / self.total_steps * 100)
    
    def compute_mape(self, predictions):
        """Compute MAPE for Pg, Qg, V against ground truth"""
        if self.ground_truth is None:
            return None, None, None
        
        # Get current values after applying action
        Pg_pred = self.network.res_gen['p_mw'].values
        Qg_pred = self.network.res_gen['q_mvar'].values
        V_pred = self.network.res_bus['vm_pu'].values
        
        # MAPE calculation
        def mape(pred, true):
            mask = np.abs(true) > 1e-6
            if not mask.any():
                return 0.0
            return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100
        
        mape_pg = mape(Pg_pred, self.ground_truth['Pg'])
        mape_qg = mape(Qg_pred, self.ground_truth['Qg'])
        mape_v = mape(V_pred, self.ground_truth['V'])
        
        return mape_pg, mape_qg, mape_v


# ============================================================================
# GRAPH ENCODER VARIANTS
# ============================================================================

class GraphMambaLayer(nn.Module):
    """Full Graph Mamba Layer with SSM and Gating (Variant A - Full PGSRL)"""
    def __init__(self, in_channels, out_channels, K=3, dropout=0.1, use_gating=True):
        super().__init__()
        self.K = K
        self.use_gating = use_gating
        
        self.lin = nn.Linear(in_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        if use_gating:
            self.gate_lin = nn.Linear(in_channels, out_channels)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.hop_weights = nn.Parameter(torch.ones(K + 1))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        if self.use_gating:
            nn.init.xavier_uniform_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)
        with torch.no_grad():
            for k in range(self.K + 1):
                self.hop_weights[k] = 1.0 / (k + 1)
    
    def forward(self, x, edge_index):
        N = x.size(0)
        device = x.device
        
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=N)
        row, col = edge_index_loop[0], edge_index_loop[1]
        
        deg = torch.zeros(N, device=device).scatter_add(
            0, row, torch.ones(row.size(0), device=device)
        ).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), norm, (N, N))
        
        hop_weights_norm = F.softmax(self.hop_weights, dim=0)
        agg = hop_weights_norm[0] * x
        x_k = x
        for k in range(1, self.K + 1):
            x_k = torch.sparse.mm(adj, x_k)
            agg = agg + hop_weights_norm[k] * x_k
        
        h = self.lin(agg)
        
        if self.use_gating:
            gate = torch.sigmoid(self.gate_lin(x))
            h = h * gate
        
        out = h + self.skip(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class StandardGCNLayer(nn.Module):
    """Standard GCN Layer (Variant B - Replace Graph Mamba with GCN)"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# ============================================================================
# ACTOR-CRITIC NETWORK VARIANTS
# ============================================================================

class Actor(nn.Module):
    """Actor network with configurable encoder"""
    def __init__(self, node_features, action_dim, encoder_type='mamba', use_gating=True):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'mamba':
            self.enc1 = GraphMambaLayer(node_features, 128, K=3, use_gating=use_gating)
            self.enc2 = GraphMambaLayer(128, 128, K=3, use_gating=use_gating)
        elif encoder_type == 'gcn':
            self.enc1 = StandardGCNLayer(node_features, 128)
            self.enc2 = StandardGCNLayer(128, 128)
        
        self.fc = nn.Linear(128, action_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.enc1(x, edge_index)
        x = self.enc2(x, edge_index)
        x = x.mean(dim=0)
        return torch.tanh(self.fc(x))


class Critic(nn.Module):
    """Critic network with configurable encoder"""
    def __init__(self, node_features, action_dim, encoder_type='mamba', use_gating=True):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'mamba':
            self.enc1 = GraphMambaLayer(node_features + action_dim, 128, K=3, use_gating=use_gating)
            self.enc2 = GraphMambaLayer(128, 128, K=3, use_gating=use_gating)
        elif encoder_type == 'gcn':
            self.enc1 = StandardGCNLayer(node_features + action_dim, 128)
            self.enc2 = StandardGCNLayer(128, 128)
        
        self.fc = nn.Linear(128, 1)
    
    def forward(self, data, action):
        x, edge_index = data.x, data.edge_index
        action_expanded = action.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, action_expanded], dim=1)
        x = self.enc1(x, edge_index)
        x = self.enc2(x, edge_index)
        x = x.mean(dim=0)
        return self.fc(x)


# ============================================================================
# AGENT VARIANTS
# ============================================================================

class TD3Agent:
    """TD3 Agent with Twin Critics (Variants A, B, C, D)"""
    def __init__(self, node_features, action_dim, encoder_type='mamba', 
                 use_gating=True, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.tau = 0.005
        self.policy_delay = 2
        self.update_count = 0
        
        # Actor
        self.actor = Actor(node_features, action_dim, encoder_type, use_gating)
        self.target_actor = Actor(node_features, action_dim, encoder_type, use_gating)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Twin Critics
        self.critic1 = Critic(node_features, action_dim, encoder_type, use_gating)
        self.critic2 = Critic(node_features, action_dim, encoder_type, use_gating)
        self.target_critic1 = Critic(node_features, action_dim, encoder_type, use_gating)
        self.target_critic2 = Critic(node_features, action_dim, encoder_type, use_gating)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
    
    def select_action(self, state, noise=0.1):
        action = self.actor(state).detach().cpu().numpy().flatten()
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -1, 1)
        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute target Q-value
        with torch.no_grad():
            next_actions = torch.vstack([self.target_actor(s) for s in next_states])
            # Add noise for target policy smoothing
            noise = torch.clamp(torch.randn_like(next_actions) * 0.2, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            target_q1 = torch.vstack([self.target_critic1(s, a) for s, a in zip(next_states, next_actions)])
            target_q2 = torch.vstack([self.target_critic2(s, a) for s, a in zip(next_states, next_actions)])
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_tensor + self.gamma * target_q * (1 - dones_tensor)
        
        # Critic loss - compute Q values for all samples
        q1_values = torch.vstack([self.critic1(s, a) for s, a in zip(states, actions_tensor)])
        q2_values = torch.vstack([self.critic2(s, a) for s, a in zip(states, actions_tensor)])
        critic_loss = F.mse_loss(q1_values, target_q) + F.mse_loss(q2_values, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Delayed policy update
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            actor_loss = -torch.mean(torch.vstack([self.critic1(s, self.actor(s)) for s in states]))
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft update targets
            self._soft_update()
        
        return critic_loss.item()
    
    def _soft_update(self):
        for target, source in [(self.target_actor, self.actor),
                               (self.target_critic1, self.critic1),
                               (self.target_critic2, self.critic2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)


class DDPGAgent:
    """DDPG Agent with Single Critic (Variant E)"""
    def __init__(self, node_features, action_dim, encoder_type='mamba',
                 use_gating=True, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.tau = 0.005
        
        # Actor
        self.actor = Actor(node_features, action_dim, encoder_type, use_gating)
        self.target_actor = Actor(node_features, action_dim, encoder_type, use_gating)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Single Critic (key difference from TD3)
        self.critic = Critic(node_features, action_dim, encoder_type, use_gating)
        self.target_critic = Critic(node_features, action_dim, encoder_type, use_gating)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
    
    def select_action(self, state, noise=0.1):
        action = self.actor(state).detach().cpu().numpy().flatten()
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -1, 1)
        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute target Q-value (single critic - more prone to overestimation)
        with torch.no_grad():
            next_actions = torch.vstack([self.target_actor(s) for s in next_states])
            target_q = torch.vstack([self.target_critic(s, a) for s, a in zip(next_states, next_actions)])
            target_q = rewards_tensor + self.gamma * target_q * (1 - dones_tensor)
        
        # Critic loss
        q_values = torch.vstack([self.critic(s, a) for s, a in zip(states, actions_tensor)])
        critic_loss = F.mse_loss(q_values, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss
        actor_loss = -torch.mean(torch.vstack([self.critic(s, self.actor(s)) for s in states]))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        self._soft_update()
        
        return critic_loss.item()
    
    def _soft_update(self):
        for target, source in [(self.target_actor, self.actor),
                               (self.target_critic, self.critic)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_agent(env, agent, episodes=100, eval_interval=10):
    """Train agent and track metrics"""
    rewards_history = []
    losses_history = []
    mape_history = {'Pg': [], 'Qg': [], 'V': []}
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action = agent.select_action(state, noise=0.2 * (1 - episode/episodes))
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            steps += 1
            
            loss = agent.train()
            episode_loss += loss
        
        rewards_history.append(total_reward)
        losses_history.append(episode_loss / max(steps, 1))
        
        # Evaluate MAPE periodically
        if (episode + 1) % eval_interval == 0:
            mape_pg, mape_qg, mape_v = env.compute_mape(None)
            if mape_pg is not None:
                mape_history['Pg'].append(mape_pg)
                mape_history['Qg'].append(mape_qg)
                mape_history['V'].append(mape_v)
    
    return rewards_history, losses_history, mape_history


def evaluate_agent(env, agent, num_episodes=20):
    """Evaluate trained agent"""
    mapes = {'Pg': [], 'Qg': [], 'V': []}
    total_violations = 0
    total_steps = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(state, noise=0)
            state, _, done, _ = env.step(action)
            steps += 1
            total_steps += 1
            
            if env._has_voltage_violation() or env._has_line_violation():
                total_violations += 1
        
        mape_pg, mape_qg, mape_v = env.compute_mape(None)
        if mape_pg is not None:
            mapes['Pg'].append(mape_pg)
            mapes['Qg'].append(mape_qg)
            mapes['V'].append(mape_v)
    
    results = {
        'MAPE_Pg': np.mean(mapes['Pg']) if mapes['Pg'] else 0,
        'MAPE_Qg': np.mean(mapes['Qg']) if mapes['Qg'] else 0,
        'MAPE_V': np.mean(mapes['V']) if mapes['V'] else 0,
        'Violation_Rate': total_violations / max(total_steps, 1) * 100
    }
    
    return results


def get_network(case_name):
    """Load IEEE test system"""
    if case_name == 'case30':
        return pn.case_ieee30()
    elif case_name == 'case118':
        return pn.case118()
    elif case_name == 'case9':
        return pn.case9()
    elif case_name == 'case39':
        return pn.case39()
    else:
        raise ValueError(f"Unknown case: {case_name}")


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================

def run_ablation_study(cases=['case30', 'case118'], episodes=100, num_runs=3):
    """
    Run complete ablation study with all five variants:
    (a) Full PGSRL
    (b) GCN variant
    (c) No Gating
    (d) No Physics-Guided Reward
    (e) Single Critic (DDPG)
    """
    
    variants = {
        'PGSRL (Full)': {
            'encoder': 'mamba',
            'use_gating': True,
            'physics_reward': True,
            'agent_type': 'td3'
        },
        'GCN Encoder': {
            'encoder': 'gcn',
            'use_gating': True,
            'physics_reward': True,
            'agent_type': 'td3'
        },
        'No Gating': {
            'encoder': 'mamba',
            'use_gating': False,
            'physics_reward': True,
            'agent_type': 'td3'
        },
        'No Physics Reward': {
            'encoder': 'mamba',
            'use_gating': True,
            'physics_reward': False,
            'agent_type': 'td3'
        },
        'Single Critic': {
            'encoder': 'mamba',
            'use_gating': True,
            'physics_reward': True,
            'agent_type': 'ddpg'
        }
    }
    
    results = {}
    convergence_data = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Testing on {case.upper()}")
        print(f"{'='*60}")
        
        results[case] = {}
        convergence_data[case] = {}
        
        for variant_name, config in variants.items():
            print(f"\n--- Variant: {variant_name} ---")
            
            run_results = []
            all_rewards = []
            
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}...", end=" ")
                
                # Create fresh environment and network
                network = get_network(case)
                env = ACOPFEnv(
                    network,
                    use_physics_reward=config['physics_reward']
                )
                
                # Create agent
                node_features = 3
                action_dim = len(network.gen) * 2
                
                if config['agent_type'] == 'td3':
                    agent = TD3Agent(
                        node_features, action_dim,
                        encoder_type=config['encoder'],
                        use_gating=config['use_gating']
                    )
                else:  # ddpg
                    agent = DDPGAgent(
                        node_features, action_dim,
                        encoder_type=config['encoder'],
                        use_gating=config['use_gating']
                    )
                
                # Train
                start_time = time.time()
                rewards, losses, mape_hist = train_agent(env, agent, episodes=episodes)
                train_time = time.time() - start_time
                
                # Evaluate
                eval_results = evaluate_agent(env, agent)
                eval_results['train_time'] = train_time
                eval_results['convergence_episode'] = find_convergence_episode(rewards)
                
                run_results.append(eval_results)
                all_rewards.append(rewards)
                
                print(f"MAPE(Pg)={eval_results['MAPE_Pg']:.2f}%, "
                      f"Viol={eval_results['Violation_Rate']:.2f}%")
            
            # Aggregate results across runs
            results[case][variant_name] = {
                'MAPE_Pg': np.mean([r['MAPE_Pg'] for r in run_results]),
                'MAPE_Pg_std': np.std([r['MAPE_Pg'] for r in run_results]),
                'MAPE_Qg': np.mean([r['MAPE_Qg'] for r in run_results]),
                'MAPE_Qg_std': np.std([r['MAPE_Qg'] for r in run_results]),
                'MAPE_V': np.mean([r['MAPE_V'] for r in run_results]),
                'MAPE_V_std': np.std([r['MAPE_V'] for r in run_results]),
                'Violation_Rate': np.mean([r['Violation_Rate'] for r in run_results]),
                'Violation_Rate_std': np.std([r['Violation_Rate'] for r in run_results]),
                'Convergence_Episode': np.mean([r['convergence_episode'] for r in run_results]),
                'Train_Time': np.mean([r['train_time'] for r in run_results])
            }
            
            # Store convergence curves
            convergence_data[case][variant_name] = np.mean(all_rewards, axis=0)
    
    return results, convergence_data


def find_convergence_episode(rewards, threshold=0.95, window=10):
    """Find episode where training converged (reached 95% of best performance)"""
    if len(rewards) < window:
        return len(rewards)
    
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    best = np.max(smoothed)
    target = best * threshold
    
    for i, r in enumerate(smoothed):
        if r >= target:
            return i + window
    return len(rewards)


def print_ablation_table(results):
    """Print ablation study results as a table"""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    
    for case, case_results in results.items():
        print(f"\n{case.upper()}")
        print("-"*100)
        print(f"{'Variant':<20} {'MAPE Pg(%)':<15} {'MAPE Qg(%)':<15} {'MAPE V(%)':<15} "
              f"{'Viol. Rate(%)':<15} {'Conv. Ep.':<12} {'Time(s)':<10}")
        print("-"*100)
        
        for variant, metrics in case_results.items():
            print(f"{variant:<20} "
                  f"{metrics['MAPE_Pg']:>6.2f}±{metrics['MAPE_Pg_std']:<6.2f} "
                  f"{metrics['MAPE_Qg']:>6.2f}±{metrics['MAPE_Qg_std']:<6.2f} "
                  f"{metrics['MAPE_V']:>6.2f}±{metrics['MAPE_V_std']:<6.2f} "
                  f"{metrics['Violation_Rate']:>6.2f}±{metrics['Violation_Rate_std']:<6.2f} "
                  f"{metrics['Convergence_Episode']:>8.1f}    "
                  f"{metrics['Train_Time']:>8.1f}")
        print("-"*100)


def generate_latex_table(results):
    """Generate LaTeX table for paper"""
    print("\n% LaTeX Table for Paper")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Ablation Study Results}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{l|ccc|c|c}")
    print("\\hline")
    print("\\textbf{Variant} & \\textbf{MAPE $P_g$ (\\%)} & \\textbf{MAPE $Q_g$ (\\%)} & "
          "\\textbf{MAPE $V$ (\\%)} & \\textbf{Viol. Rate (\\%)} & \\textbf{Conv. Ep.} \\\\")
    print("\\hline")
    
    for case, case_results in results.items():
        print(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{case.upper()}}}}} \\\\")
        print("\\hline")
        
        for variant, metrics in case_results.items():
            print(f"{variant} & "
                  f"{metrics['MAPE_Pg']:.2f}$\\pm${metrics['MAPE_Pg_std']:.2f} & "
                  f"{metrics['MAPE_Qg']:.2f}$\\pm${metrics['MAPE_Qg_std']:.2f} & "
                  f"{metrics['MAPE_V']:.2f}$\\pm${metrics['MAPE_V_std']:.2f} & "
                  f"{metrics['Violation_Rate']:.2f}$\\pm${metrics['Violation_Rate_std']:.2f} & "
                  f"{int(metrics['Convergence_Episode'])} \\\\")
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")


def plot_convergence(convergence_data, save_path='ablation_convergence.png'):
    """Plot convergence curves for all variants"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(convergence_data), figsize=(6*len(convergence_data), 5))
    if len(convergence_data) == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for ax, (case, variants) in zip(axes, convergence_data.items()):
        for (variant, rewards), color in zip(variants.items(), colors):
            # Smooth the rewards
            window = 10
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=variant, color=color, linewidth=2)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title(f'{case.upper()}', fontsize=14)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PGSRL ABLATION STUDY")
    print("For IEEE Access Paper Revision")
    print("="*60)
    
    # Run ablation study on IEEE 9-bus only for quick testing
    results, convergence_data = run_ablation_study(
        cases=['case9'],
        episodes=50,   # Quick test with fewer episodes
        num_runs=2     # Fewer runs for testing
    )
    
    # Print results
    print_ablation_table(results)
    
    # Generate LaTeX table
    generate_latex_table(results)
    
    # Plot convergence curves
    plot_convergence(convergence_data)
    
    print("\n" + "="*60)
    print("Ablation Study Complete!")
    print("="*60)
