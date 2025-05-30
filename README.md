# EV Charging Strategies

A comprehensive collection of heuristic and intelligent strategies for Electric Vehicle (EV) charging management, designed to work with the SustainGym EV charging environment.

## Overview

This repository implements various baseline strategies for EV charging environment testing, ranging from simple heuristic approaches to sophisticated COVID-19 aware strategies. These strategies serve as benchmarks for evaluating reinforcement learning algorithms in EV charging scenarios.

## Features

- **16 different charging strategies** ranging from basic to advanced
- **COVID-19 aware strategies** designed for irregular charging patterns
- **Comprehensive evaluation framework** with detailed performance metrics
- **Easy integration** with SustainGym environment
- **Extensible architecture** for adding new strategies

## Strategy Categories

### Basic Strategies

#### `RandomStrategy`
- **Purpose**: Random baseline for comparison
- **Behavior**: Generates random charging rates between 0 and 1
- **Parameters**: 
  - `seed` (optional): Random seed for reproducibility

#### `NoChargingStrategy`
- **Purpose**: Minimal baseline (never charges)
- **Behavior**: Always returns zero charging rates
- **Parameters**: None

#### `MaxChargingStrategy`
- **Purpose**: Aggressive baseline (always maximum charge)
- **Behavior**: Always charges at maximum rate (may violate constraints)
- **Parameters**: None

### Demand-Based Strategies

#### `DemandBasedStrategy`
- **Purpose**: Charges proportional to remaining demand
- **Behavior**: Normalizes charging rates based on vehicle energy requirements
- **Parameters**:
  - `max_demand_threshold` (float, default=50.0): Maximum demand for normalization

#### `UrgencyBasedStrategy`
- **Purpose**: Prioritizes vehicles leaving soon
- **Behavior**: Uses inverse of departure time to calculate urgency
- **Parameters**:
  - `urgency_threshold` (float, default=10.0): Threshold for urgency calculation

### Time-Based Strategies

#### `TimeBasedStrategy`
- **Purpose**: Adjusts charging based on time of day
- **Behavior**: Reduces charging during peak hours
- **Parameters**:
  - `peak_hours` (tuple, default=(8, 18)): Peak hour range

#### `TimeAveragedStrategy`
- **Purpose**: Distributes charging evenly over stay duration
- **Behavior**: Calculates required rate based on remaining time
- **Parameters**:
  - `max_charging_rate` (float, default=1.0): Maximum allowed charging rate

### Fixed Rate Strategies

#### `FixedRateStrategy`
- **Purpose**: Simple fixed-rate charging when vehicles present
- **Behavior**: Applies constant charging rate to all vehicles with demand
- **Parameters**:
  - `charging_rate` (float, default=0.5): Fixed charging rate (0-1)

#### `AdaptiveFixedRateStrategy`
- **Purpose**: Adapts fixed rate based on demand levels
- **Behavior**: Uses different rates for high vs low demand vehicles
- **Parameters**:
  - `base_rate` (float, default=0.5): Base charging rate
  - `high_demand_threshold` (float, default=30.0): Threshold for high demand
  - `high_demand_rate` (float, default=0.8): Rate for high demand vehicles
  - `low_demand_rate` (float, default=0.3): Rate for low demand vehicles

### Environmental Strategies

#### `MOERBasedStrategy`
- **Purpose**: Charges more when grid emissions are low
- **Behavior**: Adjusts charging based on Marginal Operating Emissions Rate
- **Parameters**:
  - `moer_threshold` (float, default=0.5): MOER threshold for clean energy

### Advanced Strategies

#### `SmartTimeBasedStrategy`
- **Purpose**: Intelligent time-based allocation considering urgency
- **Behavior**: Combines time constraints with demand-based prioritization
- **Parameters**:
  - `urgency_factor` (float, default=2.0): Weight for urgency calculation
  - `min_charging_rate` (float, default=0.1): Minimum charging rate

#### `CombinedStrategy`
- **Purpose**: Multi-factor strategy combining demand, urgency, and MOER
- **Behavior**: Weighted combination of multiple factors
- **Parameters**:
  - `demand_weight` (float, default=0.4): Weight for demand component
  - `urgency_weight` (float, default=0.4): Weight for urgency component
  - `moer_weight` (float, default=0.2): Weight for MOER component

### COVID-19 Aware Strategies

#### `ConservativeStrategy`
- **Purpose**: Stable charging during uncertain times
- **Behavior**: Conservative approach with safety margins and urgency-based adjustment
- **Parameters**:
  - `safety_margin` (float, default=0.1): Safety margin for charging rate
  - `min_demand_threshold` (float, default=1.0): Minimum demand to start charging

#### `AdaptiveLoadStrategy`
- **Purpose**: Adapts to network conditions and historical patterns
- **Behavior**: Dynamic adjustment based on MOER, time patterns, and network utilization
- **Parameters**:
  - `base_rate` (float, default=0.6): Base charging rate
  - `moer_sensitivity` (float, default=1.0): Sensitivity to MOER changes

#### `COVID19AwareStrategy`
- **Purpose**: Handles irregular patterns typical during COVID-19
- **Behavior**: Accounts for unpredictability and changed work patterns
- **Parameters**:
  - `flexibility_factor` (float, default=0.8): Base flexibility in charging
  - `uncertainty_buffer` (float, default=0.1): Buffer for uncertainty handling

## Usage

### Basic Usage

```python
from strategies import FixedRateStrategy, TimeBasedStrategy, COVID19AwareStrategy

# Create strategy instances
fixed_strategy = FixedRateStrategy(charging_rate=0.6)
time_strategy = TimeBasedStrategy(peak_hours=(9, 17))
covid_strategy = COVID19AwareStrategy(flexibility_factor=0.8)

# Use with environment
observation = env.reset()
action = strategy.get_action(observation)
```

### Environment Integration

```python
import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from strategies import ConservativeStrategy

# Create environment
gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg)

# Initialize strategy
strategy = ConservativeStrategy(safety_margin=0.1)

# Run episode
obs, info = env.reset()
strategy.reset()

while True:
    action = strategy.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Comprehensive Evaluation

```python
# Run comprehensive evaluation
python strategy_eval.py
```

This will:
- Test all strategies with 100 episodes each
- Generate performance comparison plots
- Save results to CSV file
- Compare against trained PPO model (if available)

### Custom Strategy Development

```python
from strategies import BaseStrategy
import numpy as np

class MyCustomStrategy(BaseStrategy):
    def __init__(self, my_parameter=1.0):
        super().__init__("My Custom Strategy")
        self.my_parameter = my_parameter
    
    def get_action(self, observation):
        demands = observation['demands']
        # Implement your logic here
        actions = np.where(demands > 0, self.my_parameter, 0.0)
        return actions.astype(np.float32)
    
    def reset(self):
        # Reset any internal state
        pass
```

## Evaluation Metrics

The strategies are evaluated on:

- **Mean Reward**: Average episode reward
- **Mean Profit**: Average profit from charging
- **Carbon Cost**: Environmental impact cost
- **Excess Charge**: Network violation penalties
- **Execution Time**: Strategy computation time

## Performance Results

Recent evaluation on Summer 2021 Caltech data shows:

| Strategy | Mean Reward | Mean Profit | Notes |
|----------|-------------|-------------|-------|
| Time Based | 0.15 | 0.21 | Best overall performance |
| MOER Based | 0.14 | 0.19 | Good environmental awareness |
| Random | 0.13 | 3.33 | High variance baseline |
| Conservative | 0.02 | 0.03 | Very stable, low risk |

## Requirements

- Python 3.8+
- gymnasium
- numpy
- sustaingym
- stable-baselines3 (for evaluation script)
- matplotlib, seaborn (for visualization)

## Installation

```bash
# Clone repository
git clone [your-repo-url]
cd ev-charging-strategies

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python strategy_eval.py
```

## File Structure

```
├── strategies.py          # All strategy implementations
├── strategy_eval.py       # Comprehensive evaluation script
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a new branch for your strategy
3. Implement your strategy following the `BaseStrategy` interface
4. Add tests and documentation
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use these strategies in your research, please cite:

```bibtex
@misc{ev_charging_strategies,
  title={EV Charging Strategies for SustainGym Environment},
  author={[Your Name]},
  year={2024},
  url={[Your GitHub URL]}
}
```

## Contact

[Your contact information]