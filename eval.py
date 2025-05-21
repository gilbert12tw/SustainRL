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
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')
        env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)
        env = FlattenObservation(env)
        env = Monitor(env)
        return env
    return _init

env = DummyVecEnv([make_env(seed=i) for i in range(25)])

print(env.observation_space)
print(env.action_space)

model = PPO.load('./logs/final_model/ppo_evcharging_final')

#print("評估模型...")
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
#print(f"平均獎勵: {mean_reward:.2f} +/- {std_reward:.2f}")

env = make_env()()

# 測試模型
print("測試模型表現...")
ls = []
money = []
for _ in tqdm(range(100)):
    total_return = 0
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs,deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_return += reward

    ls.append(total_return)
    money.append(info['reward_breakdown']['profit'])

print(f"測試平均獎勵: {np.mean(ls):.2f} +/- {np.std(ls):.2f}")
print(f"測試平均利潤: {np.mean(money):.2f} +/- {np.std(money):.2f}")
