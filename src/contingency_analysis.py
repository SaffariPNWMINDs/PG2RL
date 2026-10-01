"""
Reference implementation accompanying:

    Physics-Guided Graph Safe Reinforcement Learning for High-Fidelity and
    Scalable Alternating Current Optimal Power Flow
    Y. P. Singh, M. Saffari and A. Asrari, Processes, 2026.

Additional materials are available from the corresponding author
(msaffari@pnw.edu) upon reasonable request.
"""

"""
Extended N-1 Contingency Analysis for PGSRL Paper Revision

This script addresses Reviewer Comment 2:
1. N-1 contingency analysis on IEEE 118-bus system
2. Time-varying load profiles (hourly demand patterns)
3. Renewable generation uncertainty (wind/solar profiles)
4. Discussion of simulation-to-reality gap

Reference datasets:
- Load profiles: Based on typical utility load curves (IEEE RTS-96 style)
- Renewable profiles: Simulated based on NREL SAM typical meteorological year patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import defaultdict
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# LOAD AND RENEWABLE PROFILE GENERATORS
# ============================================================================

def generate_hourly_load_profile(hours=24, base_load=1.0, profile_type='residential'):
    """
    Generate realistic hourly load profiles based on typical utility patterns.
    
    Based on IEEE RTS-96 load profiles and typical utility demand curves.
    Reference: IEEE Reliability Test System (RTS-96)
    
    Args:
        hours: Number of hours to simulate
        base_load: Base load multiplier (1.0 = 100%)
        profile_type: 'residential', 'commercial', or 'industrial'
    
    Returns:
        Array of load multipliers for each hour
    """
    # Typical 24-hour load patterns (normalized to peak = 1.0)
    profiles = {
        'residential': np.array([
            0.64, 0.60, 0.58, 0.56, 0.56, 0.58,  # 00:00-05:00 (night)
            0.64, 0.76, 0.87, 0.95, 0.99, 1.00,  # 06:00-11:00 (morning rise)
            0.99, 0.93, 0.92, 0.94, 0.97, 1.00,  # 12:00-17:00 (afternoon)
            0.97, 0.96, 0.96, 0.93, 0.87, 0.75   # 18:00-23:00 (evening)
        ]),
        'commercial': np.array([
            0.45, 0.42, 0.40, 0.38, 0.38, 0.42,  # 00:00-05:00
            0.55, 0.75, 0.92, 0.98, 1.00, 1.00,  # 06:00-11:00
            0.98, 0.97, 0.98, 0.99, 1.00, 0.95,  # 12:00-17:00
            0.80, 0.65, 0.55, 0.50, 0.48, 0.46   # 18:00-23:00
        ]),
        'industrial': np.array([
            0.75, 0.73, 0.72, 0.72, 0.73, 0.78,  # 00:00-05:00
            0.88, 0.95, 0.98, 1.00, 1.00, 0.98,  # 06:00-11:00
            0.95, 0.97, 0.99, 1.00, 0.98, 0.92,  # 12:00-17:00
            0.85, 0.80, 0.78, 0.77, 0.76, 0.75   # 18:00-23:00
        ])
    }
    
    # Get base profile
    base_profile = profiles.get(profile_type, profiles['residential'])
    
    # Extend if more than 24 hours needed
    if hours > 24:
        repeats = int(np.ceil(hours / 24))
        base_profile = np.tile(base_profile, repeats)[:hours]
    else:
        base_profile = base_profile[:hours]
    
    # Add small random variations (±5%)
    noise = np.random.normal(0, 0.02, hours)
    profile = base_profile * base_load * (1 + noise)
    
    return np.clip(profile, 0.3, 1.2)


def generate_wind_profile(hours=24, capacity_factor=0.35, location='midwest'):
    """
    Generate realistic wind power profiles based on NREL wind data patterns.
    
    Based on NREL Wind Integration National Dataset (WIND) Toolkit patterns.
    Reference: https://www.nrel.gov/grid/wind-toolkit.html
    
    Args:
        hours: Number of hours
        capacity_factor: Average capacity factor (0.25-0.45 typical)
        location: 'midwest', 'texas', 'offshore' for different patterns
    
    Returns:
        Array of wind power output as fraction of capacity (0-1)
    """
    # Diurnal wind patterns vary by location
    # Midwest: stronger at night, Texas: stronger afternoon, Offshore: more stable
    patterns = {
        'midwest': np.array([
            0.45, 0.48, 0.50, 0.52, 0.50, 0.45,  # 00:00-05:00 (nighttime high)
            0.38, 0.32, 0.28, 0.25, 0.23, 0.22,  # 06:00-11:00 (morning lull)
            0.24, 0.28, 0.32, 0.35, 0.38, 0.40,  # 12:00-17:00 (afternoon rise)
            0.42, 0.44, 0.45, 0.46, 0.45, 0.44   # 18:00-23:00 (evening)
        ]),
        'texas': np.array([
            0.30, 0.28, 0.26, 0.25, 0.25, 0.28,  # 00:00-05:00
            0.32, 0.38, 0.45, 0.52, 0.58, 0.62,  # 06:00-11:00
            0.65, 0.68, 0.70, 0.68, 0.62, 0.55,  # 12:00-17:00 (afternoon peak)
            0.48, 0.42, 0.38, 0.35, 0.33, 0.31   # 18:00-23:00
        ]),
        'offshore': np.array([
            0.55, 0.54, 0.53, 0.52, 0.52, 0.53,  # More stable pattern
            0.54, 0.55, 0.56, 0.57, 0.58, 0.58,
            0.58, 0.57, 0.56, 0.55, 0.55, 0.56,
            0.57, 0.57, 0.56, 0.55, 0.55, 0.55
        ])
    }
    
    base_pattern = patterns.get(location, patterns['midwest'])
    
    # Extend pattern
    if hours > 24:
        repeats = int(np.ceil(hours / 24))
        base_pattern = np.tile(base_pattern, repeats)[:hours]
    else:
        base_pattern = base_pattern[:hours]
    
    # Scale to desired capacity factor
    base_pattern = base_pattern * (capacity_factor / np.mean(base_pattern))
    
    # Add realistic variability (wind is highly variable)
    # Use correlated noise to simulate ramping
    noise = np.zeros(hours)
    noise[0] = np.random.normal(0, 0.1)
    for i in range(1, hours):
        noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 0.1)
    
    profile = base_pattern + noise
    return np.clip(profile, 0, 1.0)


def generate_solar_profile(hours=24, capacity_factor=0.20, latitude=35):
    """
    Generate realistic solar PV profiles based on NREL SAM typical patterns.
    
    Based on NREL System Advisor Model (SAM) typical meteorological year data.
    Reference: https://sam.nrel.gov/
    
    Args:
        hours: Number of hours
        capacity_factor: Average daily capacity factor (0.15-0.25 typical)
        latitude: Location latitude (affects day length and peak)
    
    Returns:
        Array of solar power output as fraction of capacity (0-1)
    """
    # Solar follows a clear bell curve during daylight hours
    # Adjusted for latitude (higher latitude = shorter peak period)
    
    hour_of_day = np.arange(hours) % 24
    
    # Solar irradiance model (simplified clear-sky)
    sunrise = 6 - (latitude - 35) * 0.05  # Approximate sunrise
    sunset = 18 + (latitude - 35) * 0.05   # Approximate sunset
    solar_noon = 12
    
    profile = np.zeros(hours)
    for i, h in enumerate(hour_of_day):
        if sunrise <= h <= sunset:
            # Bell curve centered at solar noon
            x = (h - solar_noon) / ((sunset - sunrise) / 2)
            profile[i] = np.maximum(0, np.cos(x * np.pi / 2) ** 2)
    
    # Scale to desired capacity factor
    if np.mean(profile) > 0:
        profile = profile * (capacity_factor / np.mean(profile))
    
    # Add cloud variability (more variability during peak hours)
    cloud_factor = np.ones(hours)
    for i, h in enumerate(hour_of_day):
        if 9 <= h <= 15:  # Peak solar hours have more cloud impact
            cloud_factor[i] = np.random.uniform(0.7, 1.0)
        elif 6 <= h <= 18:
            cloud_factor[i] = np.random.uniform(0.8, 1.0)
    
    profile = profile * cloud_factor
    return np.clip(profile, 0, 1.0)


# ============================================================================
# N-1 CONTINGENCY ANALYSIS
# ============================================================================

class ContingencyAnalyzer:
    """
    Comprehensive N-1 contingency analysis for power systems.
    """
    
    def __init__(self, network):
        self.network = network
        self.base_case_results = None
        self.contingency_results = []
        
    def run_base_case(self):
        """Run base case power flow"""
        try:
            pp.runpp(self.network, algorithm='nr', max_iteration=100)
            self.base_case_results = {
                'converged': True,
                'max_vm': self.network.res_bus['vm_pu'].max(),
                'min_vm': self.network.res_bus['vm_pu'].min(),
                'max_loading': self.network.res_line['loading_percent'].max(),
                'total_loss': self.network.res_line['pl_mw'].sum()
            }
            return True
        except:
            self.base_case_results = {'converged': False}
            return False
    
    def run_n1_line_contingencies(self, verbose=True):
        """
        Run N-1 contingency analysis for all transmission lines.
        """
        results = []
        total_lines = len(self.network.line)
        
        if verbose:
            print(f"\nRunning N-1 contingency analysis for {total_lines} lines...")
        
        for idx, line_idx in enumerate(self.network.line.index):
            # Store original status
            original_status = self.network.line.at[line_idx, 'in_service']
            
            # Take line out of service
            self.network.line.at[line_idx, 'in_service'] = False
            
            result = {
                'contingency_type': 'line',
                'element_id': line_idx,
                'from_bus': self.network.line.at[line_idx, 'from_bus'],
                'to_bus': self.network.line.at[line_idx, 'to_bus']
            }
            
            try:
                pp.runpp(self.network, algorithm='nr', max_iteration=100)
                result['pf_converged'] = True
                result['max_vm'] = self.network.res_bus['vm_pu'].max()
                result['min_vm'] = self.network.res_bus['vm_pu'].min()
                result['max_loading'] = self.network.res_line['loading_percent'].max()
                result['voltage_violation'] = (result['max_vm'] > 1.1) or (result['min_vm'] < 0.9)
                result['overload'] = result['max_loading'] > 100
                result['secure'] = not (result['voltage_violation'] or result['overload'])
            except:
                result['pf_converged'] = False
                result['secure'] = False
                result['voltage_violation'] = None
                result['overload'] = None
            
            results.append(result)
            
            # Restore line
            self.network.line.at[line_idx, 'in_service'] = original_status
            
            if verbose and (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{total_lines} contingencies...")
        
        self.contingency_results = results
        return results
    
    def run_n1_generator_contingencies(self, verbose=True):
        """Run N-1 contingency analysis for generators"""
        results = []
        total_gens = len(self.network.gen)
        
        if verbose:
            print(f"\nRunning N-1 generator contingency analysis for {total_gens} generators...")
        
        for idx, gen_idx in enumerate(self.network.gen.index):
            original_status = self.network.gen.at[gen_idx, 'in_service']
            original_p = self.network.gen.at[gen_idx, 'p_mw']
            
            # Take generator out of service
            self.network.gen.at[gen_idx, 'in_service'] = False
            
            result = {
                'contingency_type': 'generator',
                'element_id': gen_idx,
                'bus': self.network.gen.at[gen_idx, 'bus'],
                'lost_mw': original_p
            }
            
            try:
                pp.runpp(self.network, algorithm='nr', max_iteration=100)
                result['pf_converged'] = True
                result['max_vm'] = self.network.res_bus['vm_pu'].max()
                result['min_vm'] = self.network.res_bus['vm_pu'].min()
                result['max_loading'] = self.network.res_line['loading_percent'].max()
                result['voltage_violation'] = (result['max_vm'] > 1.1) or (result['min_vm'] < 0.9)
                result['overload'] = result['max_loading'] > 100
                result['secure'] = not (result['voltage_violation'] or result['overload'])
            except:
                result['pf_converged'] = False
                result['secure'] = False
            
            results.append(result)
            
            # Restore generator
            self.network.gen.at[gen_idx, 'in_service'] = original_status
        
        return results
    
    def summarize_results(self):
        """Generate summary statistics"""
        if not self.contingency_results:
            return None
        
        total = len(self.contingency_results)
        converged = sum(1 for r in self.contingency_results if r['pf_converged'])
        secure = sum(1 for r in self.contingency_results if r.get('secure', False))
        voltage_viol = sum(1 for r in self.contingency_results 
                         if r.get('voltage_violation', False))
        overloads = sum(1 for r in self.contingency_results 
                       if r.get('overload', False))
        
        summary = {
            'total_contingencies': total,
            'converged': converged,
            'convergence_rate': converged / total * 100,
            'secure': secure,
            'security_rate': secure / total * 100,
            'voltage_violations': voltage_viol,
            'overloads': overloads,
            'non_converged': total - converged
        }
        
        return summary


# ============================================================================
# TIME-VARYING LOAD SIMULATION
# ============================================================================

def run_time_varying_analysis(network, hours=24, verbose=True):
    """
    Run power flow analysis with time-varying load profiles.
    """
    # Generate load profiles
    load_profile = generate_hourly_load_profile(hours, profile_type='residential')
    
    # Store original loads
    original_loads_p = network.load['p_mw'].copy()
    original_loads_q = network.load['q_mvar'].copy()
    
    results = []
    
    if verbose:
        print(f"\nRunning time-varying load analysis for {hours} hours...")
    
    for hour in range(hours):
        # Scale loads
        scale = load_profile[hour]
        network.load['p_mw'] = original_loads_p * scale
        network.load['q_mvar'] = original_loads_q * scale
        
        result = {'hour': hour, 'load_scale': scale}
        
        try:
            pp.runpp(network, algorithm='nr', max_iteration=100)
            result['converged'] = True
            result['max_vm'] = network.res_bus['vm_pu'].max()
            result['min_vm'] = network.res_bus['vm_pu'].min()
            result['max_loading'] = network.res_line['loading_percent'].max()
            result['total_gen'] = network.res_gen['p_mw'].sum()
            result['total_loss'] = network.res_line['pl_mw'].sum()
            result['voltage_violation'] = (result['max_vm'] > 1.1) or (result['min_vm'] < 0.9)
            result['overload'] = result['max_loading'] > 100
        except:
            result['converged'] = False
            result['voltage_violation'] = None
            result['overload'] = None
        
        results.append(result)
    
    # Restore original loads
    network.load['p_mw'] = original_loads_p
    network.load['q_mvar'] = original_loads_q
    
    return results, load_profile


# ============================================================================
# RENEWABLE UNCERTAINTY SIMULATION
# ============================================================================

def run_renewable_uncertainty_analysis(network, hours=24, 
                                       wind_penetration=0.20, 
                                       solar_penetration=0.10,
                                       num_scenarios=10,
                                       verbose=True):
    """
    Run analysis with renewable generation uncertainty.
    
    Args:
        network: Pandapower network
        hours: Number of hours to simulate
        wind_penetration: Wind as fraction of total generation capacity
        solar_penetration: Solar as fraction of total generation capacity
        num_scenarios: Number of Monte Carlo scenarios
    """
    # Calculate total generation capacity
    total_gen_capacity = network.gen['max_p_mw'].sum()
    wind_capacity = total_gen_capacity * wind_penetration
    solar_capacity = total_gen_capacity * solar_penetration
    
    if verbose:
        print(f"\nRenewable Uncertainty Analysis:")
        print(f"  Total generation capacity: {total_gen_capacity:.1f} MW")
        print(f"  Wind capacity ({wind_penetration*100:.0f}%): {wind_capacity:.1f} MW")
        print(f"  Solar capacity ({solar_penetration*100:.0f}%): {solar_capacity:.1f} MW")
        print(f"  Running {num_scenarios} scenarios for {hours} hours...")
    
    # Store original generator settings
    original_gen_p = network.gen['p_mw'].copy()
    original_loads_p = network.load['p_mw'].copy()
    original_loads_q = network.load['q_mvar'].copy()
    
    all_scenarios = []
    
    for scenario in range(num_scenarios):
        # Generate renewable profiles with different random seeds
        np.random.seed(42 + scenario)
        wind_profile = generate_wind_profile(hours, capacity_factor=0.35)
        solar_profile = generate_solar_profile(hours, capacity_factor=0.22)
        load_profile = generate_hourly_load_profile(hours)
        
        scenario_results = []
        
        for hour in range(hours):
            # Calculate renewable output
            wind_output = wind_capacity * wind_profile[hour]
            solar_output = solar_capacity * solar_profile[hour]
            renewable_total = wind_output + solar_output
            
            # Scale loads
            load_scale = load_profile[hour]
            network.load['p_mw'] = original_loads_p * load_scale
            network.load['q_mvar'] = original_loads_q * load_scale
            
            # Reduce conventional generation to accommodate renewables
            # (simplified dispatch - reduce all generators proportionally)
            total_load = network.load['p_mw'].sum()
            conventional_needed = max(0, total_load - renewable_total * 0.95)  # 5% losses
            
            if original_gen_p.sum() > 0:
                gen_scale = min(1.0, conventional_needed / original_gen_p.sum())
                network.gen['p_mw'] = original_gen_p * gen_scale
            
            result = {
                'scenario': scenario,
                'hour': hour,
                'wind_output': wind_output,
                'solar_output': solar_output,
                'renewable_total': renewable_total,
                'load_scale': load_scale
            }
            
            try:
                pp.runpp(network, algorithm='nr', max_iteration=100)
                result['converged'] = True
                result['max_vm'] = network.res_bus['vm_pu'].max()
                result['min_vm'] = network.res_bus['vm_pu'].min()
                result['max_loading'] = network.res_line['loading_percent'].max()
                result['voltage_violation'] = (result['max_vm'] > 1.1) or (result['min_vm'] < 0.9)
                result['overload'] = result['max_loading'] > 100
            except:
                result['converged'] = False
                result['voltage_violation'] = None
                result['overload'] = None
            
            scenario_results.append(result)
        
        all_scenarios.append(scenario_results)
        
        if verbose and (scenario + 1) % 5 == 0:
            print(f"  Completed scenario {scenario + 1}/{num_scenarios}")
    
    # Restore original settings
    network.gen['p_mw'] = original_gen_p
    network.load['p_mw'] = original_loads_p
    network.load['q_mvar'] = original_loads_q
    
    return all_scenarios, wind_profile, solar_profile


# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def plot_contingency_results(contingency_results, save_path='contingency_analysis.png'):
    """Plot N-1 contingency analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    converged = [r for r in contingency_results if r['pf_converged']]
    max_loadings = [r['max_loading'] for r in converged]
    max_vms = [r['max_vm'] for r in converged]
    min_vms = [r['min_vm'] for r in converged]
    
    # 1. Loading distribution
    ax1 = axes[0, 0]
    ax1.hist(max_loadings, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100% limit')
    ax1.set_xlabel('Maximum Line Loading (%)', fontsize=11)
    ax1.set_ylabel('Number of Contingencies', fontsize=11)
    ax1.set_title('Distribution of Maximum Line Loading', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Voltage distribution
    ax2 = axes[0, 1]
    ax2.hist(max_vms, bins=30, color='green', edgecolor='black', alpha=0.7, label='Max Vm')
    ax2.hist(min_vms, bins=30, color='orange', edgecolor='black', alpha=0.7, label='Min Vm')
    ax2.axvline(x=1.1, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=0.9, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Voltage Magnitude (p.u.)', fontsize=11)
    ax2.set_ylabel('Number of Contingencies', fontsize=11)
    ax2.set_title('Distribution of Bus Voltages', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Security status
    ax3 = axes[1, 0]
    secure = sum(1 for r in contingency_results if r.get('secure', False))
    insecure_converged = sum(1 for r in converged if not r.get('secure', True))
    non_converged = len(contingency_results) - len(converged)
    
    labels = ['Secure', 'Insecure\n(Converged)', 'Non-Converged']
    sizes = [secure, insecure_converged, non_converged]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.1)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax3.set_title('Contingency Security Status', fontsize=12)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    N-1 Contingency Analysis Summary
    ================================
    
    Total Contingencies: {len(contingency_results)}
    
    Power Flow Results:
    • Converged: {len(converged)} ({len(converged)/len(contingency_results)*100:.1f}%)
    • Non-Converged: {non_converged} ({non_converged/len(contingency_results)*100:.1f}%)
    
    Security Assessment:
    • Secure: {secure} ({secure/len(contingency_results)*100:.1f}%)
    • Voltage Violations: {sum(1 for r in converged if r.get('voltage_violation', False))}
    • Line Overloads: {sum(1 for r in converged if r.get('overload', False))}
    
    Worst Case (Converged):
    • Max Loading: {max(max_loadings):.1f}%
    • Max Voltage: {max(max_vms):.4f} p.u.
    • Min Voltage: {min(min_vms):.4f} p.u.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nContingency analysis plot saved to: {save_path}")
    plt.show()


def plot_time_varying_results(results, load_profile, save_path='time_varying_analysis.png'):
    """Plot time-varying load analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hours = [r['hour'] for r in results]
    converged = [r for r in results if r['converged']]
    
    # 1. Load profile
    ax1 = axes[0, 0]
    ax1.plot(hours, load_profile[:len(hours)], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.fill_between(hours, load_profile[:len(hours)], alpha=0.3)
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Load Multiplier', fontsize=11)
    ax1.set_title('24-Hour Load Profile', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # 2. Voltage profile
    ax2 = axes[0, 1]
    max_vms = [r['max_vm'] for r in converged]
    min_vms = [r['min_vm'] for r in converged]
    conv_hours = [r['hour'] for r in converged]
    
    ax2.plot(conv_hours, max_vms, 'r-', linewidth=2, label='Max Vm', marker='^')
    ax2.plot(conv_hours, min_vms, 'g-', linewidth=2, label='Min Vm', marker='v')
    ax2.axhline(y=1.1, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
    ax2.fill_between(conv_hours, min_vms, max_vms, alpha=0.2, color='blue')
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Voltage (p.u.)', fontsize=11)
    ax2.set_title('Voltage Profile Over 24 Hours', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    # 3. Line loading
    ax3 = axes[1, 0]
    max_loadings = [r['max_loading'] for r in converged]
    ax3.plot(conv_hours, max_loadings, 'orange', linewidth=2, marker='s')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% limit')
    ax3.fill_between(conv_hours, max_loadings, alpha=0.3, color='orange')
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Maximum Line Loading (%)', fontsize=11)
    ax3.set_title('Maximum Line Loading Over 24 Hours', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 24, 2))
    
    # 4. Generation and losses
    ax4 = axes[1, 1]
    total_gen = [r['total_gen'] for r in converged]
    total_loss = [r['total_loss'] for r in converged]
    
    ax4.plot(conv_hours, total_gen, 'b-', linewidth=2, label='Total Generation', marker='o')
    ax4.plot(conv_hours, [l * 10 for l in total_loss], 'r-', linewidth=2, 
             label='Total Losses (×10)', marker='x')
    ax4.set_xlabel('Hour of Day', fontsize=11)
    ax4.set_ylabel('Power (MW)', fontsize=11)
    ax4.set_title('Generation and Losses Over 24 Hours', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTime-varying analysis plot saved to: {save_path}")
    plt.show()


def plot_renewable_results(scenarios, wind_profile, solar_profile, 
                          save_path='renewable_uncertainty.png'):
    """Plot renewable uncertainty analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hours = range(len(wind_profile))
    
    # 1. Renewable profiles
    ax1 = axes[0, 0]
    ax1.plot(hours, wind_profile, 'b-', linewidth=2, label='Wind', marker='o', markersize=3)
    ax1.plot(hours, solar_profile, 'orange', linewidth=2, label='Solar', marker='s', markersize=3)
    ax1.fill_between(hours, wind_profile, alpha=0.3, color='blue')
    ax1.fill_between(hours, solar_profile, alpha=0.3, color='orange')
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Capacity Factor', fontsize=11)
    ax1.set_title('Renewable Generation Profiles', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # 2. Voltage spread across scenarios
    ax2 = axes[0, 1]
    
    # Collect min/max voltages per hour across scenarios
    hourly_max_vm = defaultdict(list)
    hourly_min_vm = defaultdict(list)
    
    for scenario in scenarios:
        for result in scenario:
            if result['converged']:
                hourly_max_vm[result['hour']].append(result['max_vm'])
                hourly_min_vm[result['hour']].append(result['min_vm'])
    
    hours_list = sorted(hourly_max_vm.keys())
    max_vm_mean = [np.mean(hourly_max_vm[h]) for h in hours_list]
    max_vm_std = [np.std(hourly_max_vm[h]) for h in hours_list]
    min_vm_mean = [np.mean(hourly_min_vm[h]) for h in hours_list]
    min_vm_std = [np.std(hourly_min_vm[h]) for h in hours_list]
    
    ax2.errorbar(hours_list, max_vm_mean, yerr=max_vm_std, fmt='r-', 
                 linewidth=2, capsize=3, label='Max Vm ± σ')
    ax2.errorbar(hours_list, min_vm_mean, yerr=min_vm_std, fmt='g-', 
                 linewidth=2, capsize=3, label='Min Vm ± σ')
    ax2.axhline(y=1.1, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Voltage (p.u.)', fontsize=11)
    ax2.set_title('Voltage Uncertainty Across Scenarios', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Line loading spread
    ax3 = axes[1, 0]
    
    hourly_loading = defaultdict(list)
    for scenario in scenarios:
        for result in scenario:
            if result['converged']:
                hourly_loading[result['hour']].append(result['max_loading'])
    
    loading_mean = [np.mean(hourly_loading[h]) for h in hours_list]
    loading_std = [np.std(hourly_loading[h]) for h in hours_list]
    loading_max = [np.max(hourly_loading[h]) for h in hours_list]
    
    ax3.fill_between(hours_list, 
                     [m - s for m, s in zip(loading_mean, loading_std)],
                     [m + s for m, s in zip(loading_mean, loading_std)],
                     alpha=0.3, color='orange', label='±1σ range')
    ax3.plot(hours_list, loading_mean, 'orange', linewidth=2, label='Mean')
    ax3.plot(hours_list, loading_max, 'r--', linewidth=1.5, label='Max')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Maximum Line Loading (%)', fontsize=11)
    ax3.set_title('Line Loading Uncertainty', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Convergence and violation statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_runs = sum(len(s) for s in scenarios)
    converged = sum(1 for s in scenarios for r in s if r['converged'])
    violations = sum(1 for s in scenarios for r in s 
                    if r['converged'] and (r.get('voltage_violation') or r.get('overload')))
    
    summary_text = f"""
    Renewable Uncertainty Analysis Summary
    ======================================
    
    Configuration:
    • Number of scenarios: {len(scenarios)}
    • Hours per scenario: {len(scenarios[0]) if scenarios else 0}
    • Total simulation runs: {total_runs}
    
    Results:
    • Converged: {converged} ({converged/total_runs*100:.1f}%)
    • With violations: {violations} ({violations/total_runs*100:.1f}%)
    
    Voltage Statistics (across all scenarios):
    • Mean max voltage: {np.mean([np.mean(hourly_max_vm[h]) for h in hours_list]):.4f} p.u.
    • Mean min voltage: {np.mean([np.mean(hourly_min_vm[h]) for h in hours_list]):.4f} p.u.
    
    Loading Statistics:
    • Mean max loading: {np.mean(loading_mean):.1f}%
    • Worst case loading: {max(loading_max):.1f}%
    
    Note: Profiles based on NREL-style patterns.
    Real PMU/SCADA data would improve accuracy.
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRenewable uncertainty plot saved to: {save_path}")
    plt.show()


def generate_latex_contingency_table(summary, case_name):
    """Generate LaTeX table for contingency results"""
    print(f"\n% LaTeX Table for {case_name} N-1 Contingency Analysis")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{N-1 Contingency Analysis Results for {case_name}}}")
    print(f"\\label{{tab:n1_{case_name.lower()}}}")
    print("\\begin{tabular}{l|c}")
    print("\\hline")
    print("\\textbf{Metric} & \\textbf{Value} \\\\")
    print("\\hline")
    print(f"Total Contingencies & {summary['total_contingencies']} \\\\")
    print(f"Converged & {summary['converged']} ({summary['convergence_rate']:.1f}\\%) \\\\")
    print(f"Secure & {summary['secure']} ({summary['security_rate']:.1f}\\%) \\\\")
    print(f"Voltage Violations & {summary['voltage_violations']} \\\\")
    print(f"Line Overloads & {summary['overloads']} \\\\")
    print(f"Non-Converged & {summary['non_converged']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


# ============================================================================
# SIMULATION-TO-REALITY GAP DISCUSSION
# ============================================================================

def print_sim_to_reality_discussion():
    """Print discussion of simulation-to-reality gap for paper"""
    discussion = """
================================================================================
SIMULATION-TO-REALITY GAP DISCUSSION (For Paper Limitations Section)
================================================================================

1. DATA SOURCES AND LIMITATIONS
-------------------------------
The experimental evaluation in this study relies on simulated data rather than 
real-world PMU (Phasor Measurement Unit) or SCADA (Supervisory Control and Data 
Acquisition) measurements. While the simulation environment provides a controlled 
setting for algorithm development and validation, several gaps exist between 
simulation and real-world deployment:

a) Load Profiles:
   - Used: Synthetic profiles based on IEEE RTS-96 patterns
   - Reality: Actual load patterns exhibit greater variability, including 
     weather-dependent peaks, random fluctuations, and spatial correlations
   - Gap: ±10-15% deviation from actual utility load curves expected

b) Renewable Generation:
   - Used: Simplified profiles based on NREL typical meteorological year patterns
   - Reality: Actual wind/solar output shows higher temporal resolution variability,
     forecast errors, and correlated spatial patterns across multiple sites
   - Gap: Real renewable forecast errors (15-30% for day-ahead) not captured

c) Network Parameters:
   - Used: Standard IEEE test case parameters
   - Reality: Actual line impedances vary with temperature, aging, and loading
   - Gap: Dynamic line ratings and temperature-dependent parameters not modeled

2. BRIDGING THE SIMULATION-TO-REALITY GAP
-----------------------------------------
To enhance the practical applicability of PGSRL, we recommend the following 
concrete steps for future deployment:

Step 1: Data Integration
   - Partner with utilities to obtain historical PMU/SCADA data
   - Sources: OpenPMU initiative, utility partnerships, DOE GridData program
   - Integrate NREL's Solar Power Data for Integration Studies (SPDIS) dataset
   - Use ERCOT, CAISO, or PJM publicly available operational data

Step 2: Enhanced Uncertainty Modeling
   - Implement scenario-based stochastic optimization
   - Add distributionally robust constraints for renewable uncertainty
   - Include N-k contingency analysis for critical infrastructure

Step 3: Model Validation
   - Compare PGSRL decisions against historical operator decisions
   - Conduct shadow-mode testing alongside existing EMS systems
   - Perform hardware-in-the-loop testing with RTDS/OPAL-RT simulators

Step 4: Deployment Strategy
   - Initial deployment as decision-support tool (human-in-the-loop)
   - Gradual autonomy increase based on validated performance
   - Continuous learning from operational feedback

3. ACKNOWLEDGMENT
-----------------
We acknowledge that the experimental evaluation is conducted on synthetic data 
and standard test benchmark systems. Real-world deployment would require 
validation with actual PMU/SCADA measurements and may exhibit different 
performance characteristics due to:
   - Measurement noise and bad data
   - Communication delays and missing data
   - Model-plant mismatch
   - Cyber-physical security considerations

These limitations represent important directions for future research and 
practical implementation of the proposed PGSRL framework.

================================================================================
"""
    print(discussion)
    return discussion


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("EXTENDED CONTINGENCY ANALYSIS FOR PGSRL PAPER REVISION")
    print("Addressing Reviewer Comment 2")
    print("="*70)
    
    # ========== Part 1: IEEE 118-bus N-1 Contingency Analysis ==========
    print("\n" + "="*70)
    print("PART 1: IEEE 118-BUS N-1 CONTINGENCY ANALYSIS")
    print("="*70)
    
    # Load IEEE 118-bus system
    print("\nLoading IEEE 118-bus system...")
    net118 = pn.case118()
    print(f"  Buses: {len(net118.bus)}")
    print(f"  Generators: {len(net118.gen)}")
    print(f"  Lines: {len(net118.line)}")
    print(f"  Loads: {len(net118.load)}")
    
    # Run N-1 contingency analysis
    analyzer = ContingencyAnalyzer(net118)
    
    print("\nRunning base case power flow...")
    if analyzer.run_base_case():
        print(f"  Base case converged")
        print(f"  Max voltage: {analyzer.base_case_results['max_vm']:.4f} p.u.")
        print(f"  Min voltage: {analyzer.base_case_results['min_vm']:.4f} p.u.")
        print(f"  Max loading: {analyzer.base_case_results['max_loading']:.1f}%")
    
    # Line contingencies
    line_results = analyzer.run_n1_line_contingencies(verbose=True)
    summary_118 = analyzer.summarize_results()
    
    print("\n" + "-"*50)
    print("IEEE 118-bus N-1 Contingency Summary:")
    print("-"*50)
    for key, value in summary_118.items():
        print(f"  {key}: {value}")
    
    # Generate LaTeX table
    generate_latex_contingency_table(summary_118, "IEEE 118-bus")
    
    # Plot results
    plot_contingency_results(line_results, save_path='ieee118_contingency.png')
    
    # ========== Part 2: Time-Varying Load Analysis ==========
    print("\n" + "="*70)
    print("PART 2: TIME-VARYING LOAD PROFILE ANALYSIS")
    print("="*70)
    
    # Reload network for clean state
    net118_tv = pn.case118()
    
    tv_results, load_profile = run_time_varying_analysis(net118_tv, hours=24, verbose=True)
    
    # Summary
    converged_tv = [r for r in tv_results if r['converged']]
    print(f"\n  Converged hours: {len(converged_tv)}/24")
    print(f"  Peak load hour: {np.argmax(load_profile)}")
    print(f"  Min load hour: {np.argmin(load_profile)}")
    if converged_tv:
        print(f"  Voltage range: {min(r['min_vm'] for r in converged_tv):.4f} - "
              f"{max(r['max_vm'] for r in converged_tv):.4f} p.u.")
        print(f"  Max loading range: {min(r['max_loading'] for r in converged_tv):.1f}% - "
              f"{max(r['max_loading'] for r in converged_tv):.1f}%")
    
    # Plot time-varying results
    plot_time_varying_results(tv_results, load_profile, save_path='ieee118_time_varying.png')
    
    # ========== Part 3: Renewable Uncertainty Analysis ==========
    print("\n" + "="*70)
    print("PART 3: RENEWABLE GENERATION UNCERTAINTY ANALYSIS")
    print("="*70)
    
    # Reload network
    net118_re = pn.case118()
    
    scenarios, wind_profile, solar_profile = run_renewable_uncertainty_analysis(
        net118_re,
        hours=24,
        wind_penetration=0.20,
        solar_penetration=0.10,
        num_scenarios=10,
        verbose=True
    )
    
    # Plot renewable results
    plot_renewable_results(scenarios, wind_profile, solar_profile, 
                          save_path='ieee118_renewable_uncertainty.png')
    
    # ========== Part 4: Simulation-to-Reality Gap Discussion ==========
    print("\n" + "="*70)
    print("PART 4: SIMULATION-TO-REALITY GAP")
    print("="*70)
    
    discussion = print_sim_to_reality_discussion()
    
    # Save discussion to file
    with open('sim_to_reality_discussion.txt', 'w') as f:
        f.write(discussion)
    print("\nDiscussion saved to: sim_to_reality_discussion.txt")
    
    print("\n" + "="*70)
    print("EXTENDED CONTINGENCY ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. ieee118_contingency.png - N-1 contingency analysis results")
    print("  2. ieee118_time_varying.png - Time-varying load analysis")
    print("  3. ieee118_renewable_uncertainty.png - Renewable uncertainty analysis")
    print("  4. sim_to_reality_discussion.txt - Simulation-to-reality gap discussion")


if __name__ == "__main__":
    main()
