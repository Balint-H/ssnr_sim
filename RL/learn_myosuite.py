from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from myosuite.utils import gym
from stable_baselines3 import PPO


if __name__ == '__main__':
    NUM_ENVS = 1  # Adjust based on available CPU cores
    EPISODE_LENGTH = 1000  # Max episode length

    # Function to create monitored environment instances
    def make_env():
        def _init():
            env = gym.make('myoElbowPose1D6MRandom-v0', reset_type='random')
            env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)  # Apply time limit
            env = Monitor(env)  # Monitor for logging episode rewards & lengths
            return env
        return _init

    # Create vectorized environments with monitoring
    env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)  # Ensures rollouts are logged

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy", env,
        device='cpu', verbose=1, tensorboard_log="./ppo_elbow_tensorboard/"
    )

    # Add evaluation callback for rollout logging
    eval_env = DummyVecEnv([make_env()])  # Wrap in DummyVecEnv (since it's not parallel)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Match training env
    eval_env = VecMonitor(eval_env)  # Ensures logging
    eval_callback = EvalCallback(eval_env, best_model_save_path="./elbow_gym/",
                                 log_path="./elbow_gym/", eval_freq=5_000, deterministic=True)

    # Train the model
    model.learn(600_000, progress_bar=True, callback=eval_callback)

    # Save the model
    model.save("elbow_gym")
