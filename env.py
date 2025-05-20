import numpy as np
import gymnasium as gym

class DictToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_dim = 0
        for space in self.env.observation_space.spaces.values():
            if isinstance(space, gym.spaces.Box):
                self.obs_dim += np.prod(space.shape)
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
    
    def observation(self, observation):
        if not isinstance(observation, dict):
            return observation
        flat_obs = []
        for key, value in sorted(observation.items()):
            if isinstance(value, np.ndarray):
                flat_obs.append(value.flatten())
            else:
                flat_obs.append(np.array([value], dtype=np.float32))
        return np.concatenate(flat_obs)
