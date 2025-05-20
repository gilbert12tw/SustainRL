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

from env import DictToArrayWrapper

# 創建環境
def make_env(seed=0):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')  # 使用2021訓練
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg)
        env = DictToArrayWrapper(env)
        env = Monitor(env)  # 用於記錄訓練統計數據
        return env
    return _init

# 創建評估環境
def make_eval_env(seed=1):
    def _init():
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')  # 使用2021評估
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg)
        env = DictToArrayWrapper(env)
        env = Monitor(env)
        return env
    return _init

# 創建向量化環境
env = DummyVecEnv([make_env(seed=i) for i in range(16)])  # 16個並行環境
env = VecMonitor(env, "logs/evcharging_ppo_train")

print(env.action_space)
print(env.observation_space)

# 創建評估環境
eval_env = DummyVecEnv([make_eval_env(seed=i) for i in range(2)])  # 2個評估環境
eval_env = VecMonitor(eval_env, "logs/evcharging_ppo_eval")

# 設置模型保存目錄
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# 創建回調函數
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 每10000步保存一次
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

# 創建並訓練 PPO 模型
model = PPO(
    "MlpPolicy",  # 使用 LSTM 策略
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

# 訓練模型
print("開始訓練...")
model.learn(
    total_timesteps=10000000,  # 總訓練步數
    callback=[checkpoint_callback, eval_callback],
)

# 保存最終模型
model.save(f"{log_dir}/final_model/ppo_evcharging_final")
print("訓練完成並保存模型")

# 評估訓練好的模型
print("評估模型...")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"平均獎勵: {mean_reward:.2f} +/- {std_reward:.2f}")
