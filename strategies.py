"""
Baseline strategies for EV charging environment testing.
This module implements various heuristic and simple strategies to evaluate
the performance of different approaches before applying RL algorithms.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for all charging strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset strategy state for new episode."""
        pass
    
    @abstractmethod
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action based on current observation."""
        pass


class RandomStrategy(BaseStrategy):
    """Random charging strategy - charges at random rates."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate random actions between 0 and 1."""
        num_stations = len(observation['demands'])
        return self.rng.uniform(0, 1, size=num_stations).astype(np.float32)


class NoChargingStrategy(BaseStrategy):
    """No charging strategy - never charges any vehicle."""
    
    def __init__(self):
        super().__init__("No Charging")
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Always return zero charging rates."""
        num_stations = len(observation['demands'])
        return np.zeros(num_stations, dtype=np.float32)


class MaxChargingStrategy(BaseStrategy):
    """Max charging strategy - always charges at maximum rate."""
    
    def __init__(self):
        super().__init__("Max Charging")
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Always return maximum charging rates."""
        num_stations = len(observation['demands'])
        return np.ones(num_stations, dtype=np.float32)


class DemandBasedStrategy(BaseStrategy):
    """Demand-based strategy - charges proportional to remaining demand."""
    
    def __init__(self, max_demand_threshold: float = 50.0):
        super().__init__("Demand Based")
        self.max_demand_threshold = max_demand_threshold
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Charge proportional to remaining demand."""
        demands = observation['demands']
        
        # Normalize demands to [0, 1] based on threshold
        normalized_demands = np.minimum(demands / self.max_demand_threshold, 1.0)
        
        # Only charge vehicles that have demand
        actions = np.where(demands > 0, normalized_demands, 0.0)
        
        return actions.astype(np.float32)


class TimeBasedStrategy(BaseStrategy):
    """Time-based strategy - charges more during off-peak hours."""
    
    def __init__(self, peak_hours: tuple = (8, 18)):
        super().__init__("Time Based")
        self.peak_start, self.peak_end = peak_hours
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust charging based on time of day."""
        timestep_fraction = observation['timestep'][0]  # 0 to 1
        hour_of_day = timestep_fraction * 24
        
        # Reduce charging during peak hours
        if self.peak_start <= hour_of_day < self.peak_end:
            charge_multiplier = 0.3  # 30% during peak
        else:
            charge_multiplier = 1.0  # 100% during off-peak
        
        num_stations = len(observation['demands'])
        base_action = np.ones(num_stations) * charge_multiplier
        
        # Only charge vehicles that have demand
        demands = observation['demands']
        actions = np.where(demands > 0, base_action, 0.0)
        
        return actions.astype(np.float32)


class UrgencyBasedStrategy(BaseStrategy):
    """Urgency-based strategy - prioritizes vehicles leaving soon."""
    
    def __init__(self, urgency_threshold: float = 10.0):
        super().__init__("Urgency Based")
        self.urgency_threshold = urgency_threshold
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Prioritize charging based on estimated departure time."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        
        # Calculate urgency (inverse of time to departure)
        # Add small epsilon to avoid division by zero
        time_remaining = np.maximum(est_departures, 0.1)
        urgency = 1.0 / time_remaining
        
        # Normalize urgency to [0, 1]
        if np.max(urgency) > 0:
            urgency = urgency / np.max(urgency)
        
        # Only charge vehicles that have demand
        actions = np.where(demands > 0, urgency, 0.0)
        
        return actions.astype(np.float32)


class MOERBasedStrategy(BaseStrategy):
    """MOER-based strategy - charges more when emissions are low."""
    
    def __init__(self, moer_threshold: float = 0.5):
        super().__init__("MOER Based")
        self.moer_threshold = moer_threshold
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust charging based on current and forecasted MOER."""
        demands = observation['demands']
        current_moer = observation['prev_moer'][0]
        
        # Charge more when MOER is low (cleaner energy)
        if current_moer < self.moer_threshold:
            charge_multiplier = 1.0  # Full charging when clean
        else:
            charge_multiplier = 0.5  # Reduced charging when dirty
        
        num_stations = len(demands)
        base_action = np.ones(num_stations) * charge_multiplier
        
        # Only charge vehicles that have demand
        actions = np.where(demands > 0, base_action, 0.0)
        
        return actions.astype(np.float32)


class FixedRateStrategy(BaseStrategy):
    """Fixed rate strategy - charges at a fixed rate when vehicles are present."""
    
    def __init__(self, charging_rate: float = 0.5):
        super().__init__(f"Fixed Rate ({charging_rate:.0%})")
        self.charging_rate = max(0.0, min(1.0, charging_rate))  # Clamp to [0, 1]
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Charge at fixed rate for vehicles with demand."""
        demands = observation['demands']
        
        # Use fixed charging rate for all vehicles with demand
        actions = np.where(demands > 0, self.charging_rate, 0.0)
        
        return actions.astype(np.float32)


class TimeAveragedStrategy(BaseStrategy):
    """Time-averaged strategy - distributes charging evenly over expected stay duration."""
    
    def __init__(self, max_charging_rate: float = 1.0):
        super().__init__("Time Averaged")
        self.max_charging_rate = max_charging_rate
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate charging rate based on remaining demand and time."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        
        # Calculate required charging rate to meet demand by departure
        # Each timestep represents 5 minutes, and we need to convert demand to appropriate units
        time_remaining = np.maximum(est_departures, 1.0)  # Avoid division by zero
        
        # Estimate required charging rate (normalized)
        # This is a heuristic - in practice would need to consider actual power conversion
        required_rate = np.minimum(demands / (time_remaining + 1e-6), self.max_charging_rate)
        
        # Only charge vehicles that have demand and time remaining
        actions = np.where((demands > 0) & (est_departures > 0), required_rate, 0.0)
        
        # Normalize to ensure actions are in [0, 1]
        actions = np.clip(actions, 0.0, 1.0)
        
        return actions.astype(np.float32)


class AdaptiveFixedRateStrategy(BaseStrategy):
    """Adaptive fixed rate strategy - adjusts charging rate based on demand level."""
    
    def __init__(self, base_rate: float = 0.5, high_demand_threshold: float = 30.0, 
                 high_demand_rate: float = 0.8, low_demand_rate: float = 0.3):
        super().__init__("Adaptive Fixed Rate")
        self.base_rate = base_rate
        self.high_demand_threshold = high_demand_threshold
        self.high_demand_rate = high_demand_rate
        self.low_demand_rate = low_demand_rate
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Adapt charging rate based on demand levels."""
        demands = observation['demands']
        
        # Choose charging rate based on demand level
        actions = np.zeros_like(demands)
        
        # High demand vehicles get higher charging rate
        high_demand_mask = (demands > self.high_demand_threshold) & (demands > 0)
        actions[high_demand_mask] = self.high_demand_rate
        
        # Low demand vehicles get lower charging rate
        low_demand_mask = (demands <= self.high_demand_threshold) & (demands > 0)
        actions[low_demand_mask] = self.low_demand_rate
        
        return actions.astype(np.float32)


class SmartTimeBasedStrategy(BaseStrategy):
    """Smart time-based strategy - intelligently allocates charging based on time and demand."""
    
    def __init__(self, urgency_factor: float = 2.0, min_charging_rate: float = 0.1):
        super().__init__("Smart Time Based")
        self.urgency_factor = urgency_factor
        self.min_charging_rate = min_charging_rate
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate optimal charging rate considering time constraints and demand."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        
        actions = np.zeros_like(demands)
        
        # Only process vehicles with demand and positive departure time
        active_mask = (demands > 0) & (est_departures > 0)
        
        if np.any(active_mask):
            active_demands = demands[active_mask]
            active_departures = est_departures[active_mask]
            
            # Calculate base charging rate needed to meet demand
            time_remaining = np.maximum(active_departures, 1.0)
            base_rate = active_demands / (time_remaining * 10.0)  # Scaling factor for normalization
            
            # Apply urgency factor - vehicles leaving soon get priority
            urgency_weight = 1.0 / (time_remaining ** (1.0 / self.urgency_factor))
            urgency_weight = urgency_weight / np.max(urgency_weight)  # Normalize
            
            # Combine base rate with urgency
            calculated_rate = base_rate * (1.0 + urgency_weight)
            
            # Ensure minimum charging rate for active vehicles
            calculated_rate = np.maximum(calculated_rate, self.min_charging_rate)
            
            # Clamp to valid range [0, 1]
            calculated_rate = np.clip(calculated_rate, 0.0, 1.0)
            
            actions[active_mask] = calculated_rate
        
        return actions.astype(np.float32)


class ConservativeStrategy(BaseStrategy):
    """Conservative strategy - focuses on stability and avoiding violations during uncertain times."""
    
    def __init__(self, safety_margin: float = 0.1, min_demand_threshold: float = 1.0):
        super().__init__("Conservative")
        self.safety_margin = safety_margin
        self.min_demand_threshold = min_demand_threshold
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Conservative approach focusing on stable, low-risk charging."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        
        actions = np.zeros_like(demands)
        
        # Only charge vehicles with demand (much less restrictive)
        valid_mask = demands > self.min_demand_threshold
        
        if np.any(valid_mask):
            # Use more aggressive charging rate with smaller safety margin
            base_rate = 0.6 * (1.0 - self.safety_margin)
            
            # Adjust based on urgency
            for i in range(len(demands)):
                if valid_mask[i]:
                    if est_departures[i] > 0:
                        # More urgent vehicles get higher priority
                        urgency_factor = min(10.0 / max(est_departures[i], 1.0), 2.0)
                        actions[i] = min(base_rate * urgency_factor, 0.8)
                    else:
                        actions[i] = base_rate
        
        return actions.astype(np.float32)


class AdaptiveLoadStrategy(BaseStrategy):
    """Adaptive load strategy - adjusts to current network conditions and MOER values."""
    
    def __init__(self, base_rate: float = 0.6, moer_sensitivity: float = 1.0):
        super().__init__("Adaptive Load")
        self.base_rate = base_rate
        self.moer_sensitivity = moer_sensitivity
        self.historical_demands = []
        self.step_count = 0
    
    def reset(self):
        """Reset strategy state for new episode."""
        self.historical_demands = []
        self.step_count = 0
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Adapt charging rate based on current conditions and recent history."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        current_moer = observation['prev_moer'][0]
        timestep_fraction = observation['timestep'][0]
        
        self.step_count += 1
        
        # Track demand patterns
        active_demand = np.sum(demands > 0)
        self.historical_demands.append(active_demand)
        
        # Calculate adaptive factors (less aggressive reductions)
        # MOER factor - charge more when clean energy is available
        moer_factor = 1.0 - (current_moer * self.moer_sensitivity * 0.3)  # Reduced impact
        moer_factor = np.clip(moer_factor, 0.6, 1.2)  # Less variation
        
        # Load balancing factor - less aggressive reductions
        hour_of_day = timestep_fraction * 24
        if 9 <= hour_of_day <= 17:  # Business hours during COVID
            load_factor = 0.9  # Less reduction
        elif 18 <= hour_of_day <= 22:  # Evening peak
            load_factor = 0.8  # Less reduction
        else:
            load_factor = 1.0
        
        # Network utilization factor (reduced impact)
        if len(self.historical_demands) > 5:
            recent_avg = np.mean(self.historical_demands[-5:])
            if recent_avg > len(demands) * 0.8:  # High utilization
                utilization_factor = 0.9  # Less reduction
            elif recent_avg < len(demands) * 0.2:  # Low utilization
                utilization_factor = 1.1  # Small boost
            else:
                utilization_factor = 1.0
        else:
            utilization_factor = 1.0
        
        # Calculate final charging rate
        adaptive_rate = self.base_rate * moer_factor * load_factor * utilization_factor
        adaptive_rate = np.clip(adaptive_rate, 0.3, 0.9)  # Higher minimum
        
        # Apply to vehicles with demand and add urgency factor
        actions = np.zeros_like(demands)
        for i in range(len(demands)):
            if demands[i] > 0:
                if est_departures[i] > 0:
                    urgency_factor = min(5.0 / max(est_departures[i], 1.0), 1.5)
                    actions[i] = min(adaptive_rate * urgency_factor, 0.9)
                else:
                    actions[i] = adaptive_rate
        
        return actions.astype(np.float32)


class COVID19AwareStrategy(BaseStrategy):
    """COVID-19 aware strategy - designed for irregular patterns and reduced predictability."""
    
    def __init__(self, flexibility_factor: float = 0.8, uncertainty_buffer: float = 0.1):
        super().__init__("COVID-19 Aware")
        self.flexibility_factor = flexibility_factor
        self.uncertainty_buffer = uncertainty_buffer
        self.recent_departures = []
        self.recent_demands = []
    
    def reset(self):
        """Reset strategy state for new episode."""
        self.recent_departures = []
        self.recent_demands = []
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Handle irregular charging patterns typical during COVID-19."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        timestep_fraction = observation['timestep'][0]
        
        # Track patterns for adaptation
        active_vehicles = demands > 0
        if np.any(active_vehicles):
            self.recent_departures.extend(est_departures[active_vehicles])
            self.recent_demands.extend(demands[active_vehicles])
            
            # Keep only recent history
            if len(self.recent_departures) > 20:
                self.recent_departures = self.recent_departures[-20:]
                self.recent_demands = self.recent_demands[-20:]
        
        actions = np.zeros_like(demands)
        
        # Calculate unpredictability factor (reduced impact)
        if len(self.recent_departures) > 5:
            departure_variance = np.var(self.recent_departures)
            demand_variance = np.var(self.recent_demands)
            unpredictability = min(departure_variance / 200.0 + demand_variance / 1000.0, 0.5)  # Reduced impact
        else:
            unpredictability = 0.2  # Lower default unpredictability
        
        # Adjust strategy based on time of day (COVID work patterns)
        hour_of_day = timestep_fraction * 24
        if 6 <= hour_of_day <= 10:  # Morning - potentially higher demand
            time_factor = 0.9  # Less reduction
        elif 10 <= hour_of_day <= 16:  # Midday - work from home era
            time_factor = 1.0
        elif 16 <= hour_of_day <= 20:  # Evening - uncertain patterns
            time_factor = 0.8  # Less reduction
        else:  # Night/early morning
            time_factor = 1.1  # Small boost
        
        for i in range(len(demands)):
            if demands[i] > 0:
                # Base charging rate with flexibility (higher base)
                base_rate = 0.6 * self.flexibility_factor
                
                # Adjust for uncertainty - less conservative
                uncertainty_adjustment = 1.0 - (unpredictability * self.uncertainty_buffer)
                
                # Time-based adjustment
                time_adjustment = time_factor
                
                # Urgency factor - stronger response
                if est_departures[i] > 0:
                    urgency = 1.0 / max(est_departures[i], 1.0)
                    urgency_factor = 1.0 + (urgency * 0.5)  # Stronger urgency response
                else:
                    urgency_factor = 1.0
                
                # Demand-based factor (less capped)
                demand_factor = min(demands[i] / 25.0, 1.5)  # Higher cap
                
                # Combine all factors
                final_rate = (base_rate * uncertainty_adjustment * 
                             time_adjustment * urgency_factor * demand_factor)
                
                actions[i] = np.clip(final_rate, 0.2, 0.9)  # Higher minimum, higher maximum
        
        return actions.astype(np.float32)


class CombinedStrategy(BaseStrategy):
    """Combined strategy - considers demand, urgency, and MOER."""
    
    def __init__(self, 
                 demand_weight: float = 0.4,
                 urgency_weight: float = 0.4,
                 moer_weight: float = 0.2):
        super().__init__("Combined")
        self.demand_weight = demand_weight
        self.urgency_weight = urgency_weight
        self.moer_weight = moer_weight
        
        # Ensure weights sum to 1
        total_weight = demand_weight + urgency_weight + moer_weight
        self.demand_weight /= total_weight
        self.urgency_weight /= total_weight
        self.moer_weight /= total_weight
    
    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine multiple factors for charging decision."""
        demands = observation['demands']
        est_departures = observation['est_departures']
        current_moer = observation['prev_moer'][0]
        
        # Demand component (normalized)
        demand_component = np.minimum(demands / 50.0, 1.0)
        
        # Urgency component
        time_remaining = np.maximum(est_departures, 0.1)
        urgency = 1.0 / time_remaining
        if np.max(urgency) > 0:
            urgency_component = urgency / np.max(urgency)
        else:
            urgency_component = np.zeros_like(urgency)
        
        # MOER component (charge more when MOER is low)
        moer_component = 1.0 - current_moer
        
        # Combine components
        combined_score = (self.demand_weight * demand_component +
                         self.urgency_weight * urgency_component +
                         self.moer_weight * moer_component)
        
        # Only charge vehicles that have demand
        actions = np.where(demands > 0, combined_score, 0.0)
        
        return actions.astype(np.float32)


def test_strategy(env: gym.Env, strategy: BaseStrategy, num_episodes: int = 1, verbose: bool = True) -> Dict[str, Any]:
    """
    Test a strategy on the environment and return performance metrics.
    
    Args:
        env: The environment to test on
        strategy: The strategy to test
        num_episodes: Number of episodes to run
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing performance metrics
    """
    episode_rewards = []
    episode_info = []
    
    for episode in range(num_episodes):
        strategy.reset()
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        if verbose:
            print(f"\n=== Testing {strategy.name} - Episode {episode + 1} ===")
        
        while True:
            action = strategy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_info.append(info)
        
        if verbose:
            print(f"Episode {episode + 1} completed:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Reward breakdown: {info.get('reward_breakdown', {})}")
            if 'max_profit' in info:
                efficiency = (info['reward_breakdown']['profit'] / info['max_profit']) * 100
                print(f"  Profit efficiency: {efficiency:.2f}%")
    
    # Calculate summary statistics
    results = {
        'strategy_name': strategy.name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_info': episode_info
    }
    
    if episode_info:
        # Average reward breakdown across episodes
        avg_breakdown = {}
        for key in episode_info[0]['reward_breakdown'].keys():
            avg_breakdown[key] = np.mean([info['reward_breakdown'][key] for info in episode_info])
        results['avg_reward_breakdown'] = avg_breakdown
        
        # Average other metrics
        if 'max_profit' in episode_info[0]:
            results['avg_max_profit'] = np.mean([info['max_profit'] for info in episode_info])
            results['avg_profit_efficiency'] = (avg_breakdown['profit'] / results['avg_max_profit']) * 100
    
    return results 