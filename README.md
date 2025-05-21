# SustainRL

## Requirements

- Python 3.10

## Installation

1. Install the sustainable EV charging environment:
   ```
   pip install 'sustaingym[ev]'
   ```

2. Install other dependencies:
   ```
   pip install -r requirement.txt
   ```

## Project Structure

- `train.py`: Main training script with Hydra configuration support
- `conf/config.yaml`: Configuration file with parameters for all algorithms
- `eval.py`: Evaluation script for trained models

## Supported Algorithms

The project supports multiple reinforcement learning algorithms:

- **PPO** (Proximal Policy Optimization): On-policy algorithm with good sample efficiency
- **RPPO** (Recurrent PPO): Extension of PPO with recurrent networks for partially observable environments
- **SAC** (Soft Actor-Critic): Off-policy actor-critic algorithm with entropy maximization
- **TD3** (Twin Delayed DDPG): Off-policy algorithm that addresses overestimation bias in actor-critic methods
- **DDPG** (Deep Deterministic Policy Gradient): Off-policy algorithm for continuous action spaces
- **ARS** (Augmented Random Search): Simple yet effective random search method
- **TQC** (Truncated Quantile Critics): Off-policy algorithm that learns a distributional value function

## Usage

### Training with Default Settings

```python
python train.py
```


This will train a PPO agent with the default parameters specified in `conf/config.yaml`.

### Selecting Different Algorithms

To train with a different algorithm:

```python
python train.py algo=SAC
```


Available options: `PPO`, `RPPO`, `SAC`, `TD3`, `DDPG`, `ARS`, `TQC`

### Customizing Parameters

You can override any parameter defined in the config file:

```python
python train.py algo=SAC SAC.learning_rate=0.0001 SAC.batch_size=512
```

### Changing Training Settings

Modify general training parameters:

```python
python train.py timesteps=5000000 seed=42 num_envs=8
```


### Evaluation

After training, models are automatically evaluated. You can also use `eval.py` to evaluate saved models:

```python
python eval.py --model_path logs/final_model/ppo_evcharging_final
```


## Monitoring Training

Training progress can be monitored through TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```