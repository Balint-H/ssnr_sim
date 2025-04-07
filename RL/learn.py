import gymnasium as gym
from cartpole import swingup, _DEFAULT_TIME_LIMIT

from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import TimeLimit

import os

TENSORBOARD_LOG_DIR = "./ppo_cartpole_tensorboard/"

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        print(f"Creating environment in process {rank}...")
        # 1. Instantiate the dm_control environmen
        dm_control_env = swingup(random=seed + rank) # Pass distinct random seeds

        # 2. Wrap the dm_control environment using Shimmy (so its gym format)
        env = DmControlCompatibilityV0(dm_control_env)

        # 3. Apply TimeLimit
        timestep = dm_control_env.physics.model.opt.timestep
        max_steps = int(_DEFAULT_TIME_LIMIT / timestep)
        env = TimeLimit(env, max_episode_steps=max_steps)

        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    num_cpu = 4  
    env_seed = 42
    
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    print(f"Setting up {num_cpu} parallel environments...")
    # Vectorized environments run in parallel to collect experience (here on cpu)
    vec_env = SubprocVecEnv([make_env(i, env_seed) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env)  # log episode reward stats

    # Instantiate the PPO agent
    print("Instantiating SB3 PPO agent...")
    # MultiInputPolicy is an MLP that flattens the dict of observations to an input
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=200000, progress_bar=True) # Adjust timesteps as needed
    print("Training finished.")

    model.save("ppo_dm_cartpole_local_parallel")
    print("Model saved.")

    vec_env.close()