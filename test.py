import gymnasium as gym
from sustaingym.envs.evcharging import GMMsTraceGenerator
from sustaingym.envs.evcharging import DiscreteActionWrapper
from tqdm import tqdm
import numpy as np

gmmg = GMMsTraceGenerator('caltech', 'Summer 2021')
env = gym.make('sustaingym/EVCharging-v0', data_generator=gmmg, project_action_in_env=False)

print(env.action_space)
print(env.spec.max_episode_steps)

ls = []
money = []

for _ in tqdm(range(15)):
    total_return = 0
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_return += reward
    print(steps)
    ls.append(total_return)
    money.append(info['reward_breakdown']['profit'])
    #    print(reward)
    #    print(f"Profit: {info['reward_breakdown']['profit']}, Carbon: {info['reward_breakdown']['carbon_cost']}, Excess: {info['reward_breakdown']['excess_charge']}")

print(np.mean(ls), np.std(ls))
print(np.mean(money), np.std(money))
