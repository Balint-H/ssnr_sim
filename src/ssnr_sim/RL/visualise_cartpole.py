import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
import time


# Load the policy globally so it's accessible by the control callback
policy_path = "ppo_dm_cartpole_local_parallel.zip" 
policy = None
try:
    policy = PPO.load(policy_path, device='cpu')
    print(f"Loaded policy from {policy_path}")
except Exception as e:
    print(f"Error loading policy from {policy_path}: {e}")
    print("Please ensure the policy file exists.")
    # We don't exit here, viewer might still launch without policy


# Reimplement the observations from cartpole for classic mujoco
def get_observation(data: mujoco.MjData) -> dict:
    cart_pos = data.joint('slider').qpos[0]
    pole_cos = np.cos(data.joint('hinge_1').qpos[0])
    pole_sin = np.sin(data.joint('hinge_1').qpos[0])
    position_obs = np.array([cart_pos, pole_cos, pole_sin])

    cart_vel = data.joint('slider').qvel[0]
    pole_vel = data.joint('hinge_1').qvel[0]
    velocity_obs = np.array([cart_vel, pole_vel])

    obs = {
        "position": position_obs.astype(np.float32),
        "velocity": velocity_obs.astype(np.float32)
    }
    return obs



def control_callback(model, data):
    if policy is None:
        # No policy loaded, do nothing
        return

    observation = get_observation(data)

    action, _ = policy.predict(observation, deterministic=True)

    data.ctrl[:] = action


def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)
    print("Loading MuJoCo model...")
    
    model = mujoco.MjModel.from_xml_path("cartpole.xml")
    data = mujoco.MjData(model)

    if model is not None:
        print("Model loaded. Setting control callback.")
        mujoco.set_mjcb_control(control_callback)
    else:
        print("Failed to load model.")

    return model, data


if __name__ == '__main__':
    mujoco.viewer.launch(loader=load_callback)