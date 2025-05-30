"""
Comprehensive strategy evaluation script for EV charging environment.
This script tests all implemented strategies using the same environment setup as eval.py
and compares them against the trained PPO model.
"""

import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Import our strategies
from strategies import (
    RandomStrategy, NoChargingStrategy, MaxChargingStrategy,
    DemandBasedStrategy, TimeBasedStrategy, UrgencyBasedStrategy,
    MOERBasedStrategy, CombinedStrategy, BaseStrategy,
    FixedRateStrategy, TimeAveragedStrategy, AdaptiveFixedRateStrategy,
    SmartTimeBasedStrategy, ConservativeStrategy, AdaptiveLoadStrategy,
    COVID19AwareStrategy
)


class FlattenObservationStrategyWrapper:
    """
    Wrapper to adapt our dictionary-based strategies to work with FlattenObservation.
    This converts flattened observations back to dictionary format for our strategies.
    """
    
    def __init__(self, strategy: BaseStrategy, observation_space):
        self.strategy = strategy
        self.observation_space = observation_space
        
        # Determine the structure of the original observation space
        # Based on the EVCharging environment structure
        self.obs_mapping = self._create_observation_mapping()
    
    def _create_observation_mapping(self):
        """Create mapping from flattened observation to dictionary structure."""
        # Standard EVCharging observation structure (from environment code)
        # This assumes the observation space follows the standard structure
        mapping = {
            'timestep': (0, 1),           # 1 element
            'est_departures': None,       # Will be determined by environment
            'demands': None,              # Will be determined by environment  
            'prev_moer': None,            # 1 element
            'forecasted_moer': None       # moer_forecast_steps elements (default 36)
        }
        
        # Try to infer structure from observation space shape
        total_size = self.observation_space.shape[0]
        
        # Common structure for EVCharging environment:
        # timestep(1) + est_departures(n) + demands(n) + prev_moer(1) + forecasted_moer(36)
        # where n is number of stations
        
        # Calculate number of stations (assuming 36 forecast steps)
        moer_forecast_steps = 36
        fixed_size = 1 + 1 + moer_forecast_steps  # timestep + prev_moer + forecasted_moer
        num_stations = (total_size - fixed_size) // 2  # est_departures + demands
        
        current_idx = 0
        mapping['timestep'] = (current_idx, current_idx + 1)
        current_idx += 1
        
        mapping['est_departures'] = (current_idx, current_idx + num_stations)
        current_idx += num_stations
        
        mapping['demands'] = (current_idx, current_idx + num_stations)
        current_idx += num_stations
        
        mapping['prev_moer'] = (current_idx, current_idx + 1)
        current_idx += 1
        
        mapping['forecasted_moer'] = (current_idx, current_idx + moer_forecast_steps)
        
        return mapping
    
    def _flatten_to_dict(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flattened observation back to dictionary format."""
        obs_dict = {}
        for key, (start, end) in self.obs_mapping.items():
            obs_dict[key] = obs[start:end].astype(np.float32)
        return obs_dict
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from strategy using flattened observation."""
        obs_dict = self._flatten_to_dict(obs)
        return self.strategy.get_action(obs_dict)
    
    def reset(self):
        """Reset the wrapped strategy."""
        self.strategy.reset()
    
    @property
    def name(self):
        """Get strategy name."""
        return self.strategy.name


def make_env(seed=0, verbose=False):
    """Create environment following eval.py setup."""
    gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
    env = gym.make('sustaingym/EVCharging-v0', 
                   data_generator=gmmg, 
                   project_action_in_env=False,
                   verbose=0 if not verbose else 1)
    env = FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def load_trained_model(model_path: str) -> Optional[PPO]:
    """Load trained PPO model if available."""
    try:
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            print(f"Successfully loaded trained model from {model_path}")
            return model
        else:
            print(f"Model not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_strategy_performance(strategy_wrapper: FlattenObservationStrategyWrapper, 
                            env: gym.Env, 
                            num_episodes: int = 100,
                            verbose: bool = False) -> Dict[str, Any]:
    """
    Test strategy performance following eval.py methodology.
    
    Args:
        strategy_wrapper: Wrapped strategy to test
        env: Environment to test on
        num_episodes: Number of episodes to run
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing performance metrics
    """
    total_rewards = []
    total_profits = []
    episode_lengths = []
    reward_breakdowns = []
    
    iterator = tqdm(range(num_episodes), desc=f"Testing {strategy_wrapper.name}") if verbose else range(num_episodes)
    
    for episode in iterator:
        strategy_wrapper.reset()
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action = strategy_wrapper.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract profit and reward breakdown if available
        if 'reward_breakdown' in info:
            total_profits.append(info['reward_breakdown']['profit'])
            reward_breakdowns.append(info['reward_breakdown'])
        else:
            total_profits.append(0.0)
            reward_breakdowns.append({'profit': 0.0, 'carbon_cost': 0.0, 'excess_charge': 0.0})
    
    # Calculate statistics
    results = {
        'strategy_name': strategy_wrapper.name,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_profit': np.mean(total_profits),
        'std_profit': np.std(total_profits),
        'mean_episode_length': np.mean(episode_lengths),
        'total_rewards': total_rewards,
        'total_profits': total_profits,
        'reward_breakdowns': reward_breakdowns
    }
    
    # Average reward breakdown
    if reward_breakdowns:
        avg_breakdown = {}
        for key in reward_breakdowns[0].keys():
            avg_breakdown[key] = np.mean([rb[key] for rb in reward_breakdowns])
        results['avg_reward_breakdown'] = avg_breakdown
    
    return results


def test_ppo_model(model: PPO, env: gym.Env, num_episodes: int = 100) -> Dict[str, Any]:
    """Test PPO model performance following eval.py methodology."""
    total_rewards = []
    total_profits = []
    episode_lengths = []
    reward_breakdowns = []
    
    print("Testing PPO model...")
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if 'reward_breakdown' in info:
            total_profits.append(info['reward_breakdown']['profit'])
            reward_breakdowns.append(info['reward_breakdown'])
        else:
            total_profits.append(0.0)
            reward_breakdowns.append({'profit': 0.0, 'carbon_cost': 0.0, 'excess_charge': 0.0})
    
    # Calculate statistics
    results = {
        'strategy_name': 'PPO (Trained)',
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_profit': np.mean(total_profits),
        'std_profit': np.std(total_profits),
        'mean_episode_length': np.mean(episode_lengths),
        'total_rewards': total_rewards,
        'total_profits': total_profits,
        'reward_breakdowns': reward_breakdowns
    }
    
    # Average reward breakdown
    if reward_breakdowns:
        avg_breakdown = {}
        for key in reward_breakdowns[0].keys():
            avg_breakdown[key] = np.mean([rb[key] for rb in reward_breakdowns])
        results['avg_reward_breakdown'] = avg_breakdown
    
    return results


def create_comprehensive_comparison(results: List[Dict[str, Any]], save_path: str = "strategy_comparison_comprehensive.png"):
    """Create comprehensive comparison plots."""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EV Charging Strategy Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # Extract data
    strategy_names = [r['strategy_name'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    mean_profits = [r['mean_profit'] for r in results]
    std_profits = [r['std_profit'] for r in results]
    
    # Plot 1: Mean Rewards
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(strategy_names)), mean_rewards, yerr=std_rewards, 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('Mean Episode Rewards', fontweight='bold')
    ax1.set_ylabel('Reward ($)')
    ax1.set_xticks(range(len(strategy_names)))
    ax1.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, reward, std) in enumerate(zip(bars1, mean_rewards, std_rewards)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std, 
                f'{reward:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Mean Profits
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(strategy_names)), mean_profits, yerr=std_profits,
                    capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.set_title('Mean Profits', fontweight='bold')
    ax2.set_ylabel('Profit ($)')
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, profit, std) in enumerate(zip(bars2, mean_profits, std_profits)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std, 
                f'{profit:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Reward Breakdown (if available)
    ax3 = axes[0, 2]
    if all('avg_reward_breakdown' in r for r in results):
        profit_values = [r['avg_reward_breakdown']['profit'] for r in results]
        carbon_costs = [r['avg_reward_breakdown']['carbon_cost'] for r in results]
        excess_charges = [r['avg_reward_breakdown']['excess_charge'] for r in results]
        
        x_pos = np.arange(len(strategy_names))
        ax3.bar(x_pos, profit_values, label='Profit', alpha=0.8, color='green')
        ax3.bar(x_pos, [-c for c in carbon_costs], label='Carbon Cost', alpha=0.8, color='orange')
        ax3.bar(x_pos, [-e for e in excess_charges], label='Excess Charge', alpha=0.8, color='red')
        
        ax3.set_title('Reward Components', fontweight='bold')
        ax3.set_ylabel('Component Value ($)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution (Box Plot)
    ax4 = axes[1, 0]
    reward_data = [r['total_rewards'] for r in results]
    ax4.boxplot(reward_data, labels=strategy_names)
    ax4.set_title('Reward Distribution', fontweight='bold')
    ax4.set_ylabel('Episode Reward ($)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Profit Distribution (Box Plot)
    ax5 = axes[1, 1]
    profit_data = [r['total_profits'] for r in results]
    ax5.boxplot(profit_data, labels=strategy_names)
    ax5.set_title('Profit Distribution', fontweight='bold')
    ax5.set_ylabel('Episode Profit ($)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance Ranking
    ax6 = axes[1, 2]
    # Sort strategies by mean reward for ranking
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    sorted_names = [r['strategy_name'] for r in sorted_results]
    sorted_rewards = [r['mean_reward'] for r in sorted_results]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
    bars6 = ax6.barh(range(len(sorted_names)), sorted_rewards, color=colors, alpha=0.8)
    ax6.set_title('Performance Ranking', fontweight='bold')
    ax6.set_xlabel('Mean Reward ($)')
    ax6.set_yticks(range(len(sorted_names)))
    ax6.set_yticklabels(sorted_names)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, reward) in enumerate(zip(bars6, sorted_rewards)):
        ax6.text(bar.get_width() + max(sorted_rewards) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{reward:.1f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved to {save_path}")
    plt.show()


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comprehensive summary DataFrame."""
    summary_data = []
    
    for result in results:
        row = {
            'Strategy': result['strategy_name'],
            'Mean Reward': result['mean_reward'],
            'Std Reward': result['std_reward'],
            'Mean Profit': result['mean_profit'],
            'Std Profit': result['std_profit'],
            'Mean Episode Length': result['mean_episode_length']
        }
        
        if 'avg_reward_breakdown' in result:
            breakdown = result['avg_reward_breakdown']
            row['Avg Profit Component'] = breakdown['profit']
            row['Avg Carbon Cost'] = breakdown['carbon_cost']
            row['Avg Excess Charge'] = breakdown['excess_charge']
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Sort by mean reward
    df = df.sort_values('Mean Reward', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Start ranking from 1
    
    return df


def main():
    """Main function to run comprehensive strategy evaluation."""
    print("=" * 80)
    print("COMPREHENSIVE EV CHARGING STRATEGY EVALUATION")
    print("=" * 80)
    
    # Parameters
    num_episodes = 100
    seed = 42
    model_path = './logs/final_model/ppo_evcharging_final'
    
    # Create environment
    print("Creating environment...")
    env = make_env(seed=seed, verbose=False)
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Initialize strategies
    base_strategies = [
        RandomStrategy(seed=seed),
        NoChargingStrategy(),
        MaxChargingStrategy(),
        DemandBasedStrategy(max_demand_threshold=50.0),
        TimeBasedStrategy(peak_hours=(8, 18)),
        UrgencyBasedStrategy(),
        MOERBasedStrategy(moer_threshold=0.5),
        CombinedStrategy(demand_weight=0.1, urgency_weight=0.2, moer_weight=0.7),
        # New strategies
        FixedRateStrategy(charging_rate=0.5),  # 50% fixed rate
        FixedRateStrategy(charging_rate=0.7),  # 70% fixed rate
        TimeAveragedStrategy(max_charging_rate=1.0),
        AdaptiveFixedRateStrategy(
            base_rate=0.5, 
            high_demand_threshold=25.0,
            high_demand_rate=0.8,
            low_demand_rate=0.3
        ),
        SmartTimeBasedStrategy(urgency_factor=2.0, min_charging_rate=0.1),
        # COVID-19 aware strategies (optimized parameters)
        ConservativeStrategy(safety_margin=0.1, min_demand_threshold=1.0),
        AdaptiveLoadStrategy(base_rate=0.6, moer_sensitivity=1.0),
        COVID19AwareStrategy(flexibility_factor=0.8, uncertainty_buffer=0.1)
    ]
    
    # Wrap strategies for flattened observations
    print(f"\nWrapping {len(base_strategies)} strategies for flattened observations...")
    wrapped_strategies = [
        FlattenObservationStrategyWrapper(strategy, env.observation_space) 
        for strategy in base_strategies
    ]
    
    # Test all strategies
    print(f"\nTesting strategies with {num_episodes} episodes each...")
    results = []
    
    for wrapped_strategy in wrapped_strategies:
        print(f"\nTesting {wrapped_strategy.name}...")
        start_time = time.time()
        
        result = test_strategy_performance(
            wrapped_strategy, env, num_episodes=num_episodes, verbose=True
        )
        
        end_time = time.time()
        result['execution_time'] = end_time - start_time
        
        results.append(result)
        
        print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Mean profit: {result['mean_profit']:.2f} ± {result['std_profit']:.2f}")
        print(f"  Execution time: {result['execution_time']:.2f}s")
    
    # Test trained PPO model if available
    ppo_model = load_trained_model(model_path)
    if ppo_model is not None:
        print("\nTesting trained PPO model...")
        ppo_result = test_ppo_model(ppo_model, env, num_episodes=num_episodes)
        results.append(ppo_result)
        print(f"  PPO Mean reward: {ppo_result['mean_reward']:.2f} ± {ppo_result['std_reward']:.2f}")
        print(f"  PPO Mean profit: {ppo_result['mean_profit']:.2f} ± {ppo_result['std_profit']:.2f}")
    
    # Create summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    summary_df = create_summary_dataframe(results)
    print(summary_df.round(2).to_string())
    
    # Save results
    summary_df.to_csv('strategy_evaluation_results.csv', index=True)
    print(f"\nResults saved to 'strategy_evaluation_results.csv'")
    
    # Create visualizations
    print("\nGenerating comprehensive comparison plots...")
    try:
        create_comprehensive_comparison(results)
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Find best strategy
    best_strategy = max(results, key=lambda x: x['mean_reward'])
    print(f"\n🏆 Best performing strategy: {best_strategy['strategy_name']}")
    print(f"   Mean reward: {best_strategy['mean_reward']:.2f} ± {best_strategy['std_reward']:.2f}")
    print(f"   Mean profit: {best_strategy['mean_profit']:.2f} ± {best_strategy['std_profit']:.2f}")
    
    # Show improvement over random baseline
    random_result = next((r for r in results if 'Random' in r['strategy_name']), None)
    if random_result:
        improvement = ((best_strategy['mean_reward'] - random_result['mean_reward']) / 
                      abs(random_result['mean_reward']) * 100)
        print(f"   Improvement over random: {improvement:.1f}%")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main() 