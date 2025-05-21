import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from sustaingym.envs.evcharging import DiscreteActionWrapper
from tqdm import tqdm
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation

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

env = DummyVecEnv([make_env(seed=i) for i in range(16)])
env = VecMonitor(env, "logs/evcharging_ppo_train")

print(env.observation_space)
print(env.action_space)

eval_env = DummyVecEnv([make_eval_env(seed=i) for i in range(2)])
eval_env = VecMonitor(eval_env, "logs/evcharging_ppo_eval")

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path=f"{log_dir}/models/",
    name_prefix="ppo_evcharging",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{log_dir}/best_model/",
    log_path=f"{log_dir}/eval_results/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,  # 設置學習率
    n_steps=2048,        # 每次更新前收集的步數
    batch_size=64,       # 訓練時的批次大小
    n_epochs=10,         # 每次更新時的迭代次數
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE係數
    clip_range=0.2,      # 策略梯度裁剪範圍
    ent_coef=0.01,       # 熵係數，鼓勵探索
    vf_coef=0.5,         # 值函數係數
    max_grad_norm=0.5,   # 梯度裁剪
    verbose=1,
    tensorboard_log=f"{log_dir}/tensorboard/",
)

print("開始訓練...")
model.learn(
    total_timesteps=10000000,
    callback=[checkpoint_callback, eval_callback],
)

model.save(f"{log_dir}/final_model/ppo_evcharging_final")
print("訓練完成並保存模型")

print("評估模型...")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"平均獎勵: {mean_reward:.2f} +/- {std_reward:.2f}")
