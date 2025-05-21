import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from gymnasium.wrappers import FlattenObservation
import numpy as np
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Import RL algorithms
from stable_baselines3 import PPO, SAC, TD3, DDPG
from sb3_contrib import RecurrentPPO, ARS, TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def make_env(seed=0):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        env = Monitor(env)
        return env
    return _init

def make_eval_env(seed=1):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        env = Monitor(env)
        return env
    return _init

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    np.random.seed(cfg.seed)
    
    # Get algorithm parameters
    algo_name = cfg.algo
    algo_params = dict(cfg[algo_name])
    
    # Prepare directories
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environments
    env = DummyVecEnv([make_env(seed=i) for i in range(cfg.num_envs)])
    env = VecMonitor(env, f"{log_dir}/{algo_name.lower()}_evcharging_train")
    
    eval_env = DummyVecEnv([make_eval_env(seed=i) for i in range(cfg.num_eval_envs)])
    eval_env = VecMonitor(eval_env, f"{log_dir}/{algo_name.lower()}_evcharging_eval")
    
    # Print environment info
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=f"{log_dir}/models/",
        name_prefix=f"{algo_name.lower()}_evcharging",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model/",
        log_path=f"{log_dir}/eval_results/",
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes_during_training,
        deterministic=True,
        render=False,
    )
    
    # Initialize the appropriate model based on algo
    print(f"Initializing {algo_name} with parameters: {algo_params}")
    
    # Default policy for most algorithms
    policy = "MlpPolicy"
    
    # Create the appropriate model based on algorithm name
    if algo_name == 'PPO':
        model = PPO(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'RPPO':
        model = RecurrentPPO(
            "MlpLstmPolicy",  # RecurrentPPO uses LSTM policy
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'SAC':
        model = SAC(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'TD3':
        model = TD3(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'DDPG':
        model = DDPG(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'ARS':
        model = ARS(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    elif algo_name == 'TQC':
        model = TQC(
            policy,
            env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard/",
            **algo_params
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    
    # Train the model
    print(f"Starting training with {algo_name}...")
    model.learn(
        total_timesteps=cfg.timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    
    # Save the final model
    final_model_path = f"{log_dir}/final_model/{algo_name.lower()}_evcharging_final"
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")
    
    # Evaluate the trained model
    print("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.eval_episodes, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()
