import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from loop_rate_limiters import RateLimiter
from learn_myosuite import make_env
from myosuite.utils import gym
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0

# Visualize an MJX environment interactively; no policy

def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = PPO.load("./elbow_gym/best_model.zip", env=env)
    base_env = env.unwrapped.envs[0].unwrapped
    m, d = get_mj_model_data(base_env)

    obs = env.reset()

    rate = RateLimiter(frequency=1/base_env.sim.model.opt.timestep//20, warn=False)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():

            obs, reward, done, info = env.step(model.predict(obs, deterministic=True)[0])

            d.qpos = base_env.sim.data.qpos
            mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.
            # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
            d.sensordata[0] = reward[0]

            # And update ctrl to see muscle activations
            d.ctrl = base_env.sim.data.ctrl

            # Update target location - You might see an offset in what the agent reaches... try rerunning the script
            # a few times and check again. Look into the reset function of PoseEnvV0 (see imports) to see why!
            m.site_pos[base_env.target_sids[0]] = base_env.sim.model.site_pos[base_env.target_sids[0]]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # Sleep so we run in real time
            rate.sleep()
    pass


def get_mj_model_data(base_env):
    # We could get the model from the env, but we want to make some edits for convenience
    spec = mujoco.MjSpec.from_file(gym.registry['myoElbowPose1D6MRandom-v0'].kwargs['model_path'])
    # Add in dummy sensor we can write to later to visualize values
    spec.add_sensor(name="reward", type=mujoco.mjtSensor.mjSENS_USER, dim=1)

    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


if __name__ == '__main__':
    main()
