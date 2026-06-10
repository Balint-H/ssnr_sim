import mujoco
import jax
import numpy as np
import mujoco.viewer
import pickle
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
from loop_rate_limiters import rate_limiter


from planar_arm_env import PlanarArmTendon, default_config


def main():
  # Instantiate the environment
  env = PlanarArmTendon()

  # We use Brax's trick of setting num_timesteps=0 to build the exact network
  # and observation pipeline instantly.
  ppo_params = {
    "num_timesteps": 0,
    "num_envs": 1,
    "unroll_length": 20,
    "discounting": 0.99,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "num_minibatches": 32,
    "entropy_cost": 1e-3,
    "num_evals": 1,
    "seed": 0,
    "episode_length": default_config.episode_length,
    "normalize_observations": True,
  }

  network_params = {
    "policy_hidden_layer_sizes": (64, 64),
    "value_hidden_layer_sizes": (64, 64),
  }

  network_factory = functools.partial(ppo_networks.make_ppo_networks, **network_params)

  train_fn = functools.partial(
    ppo.train,
    **ppo_params,
    network_factory=network_factory,
  )

  print("Reconstructing network architecture and observation normalizer...")
  make_inference_fn, _, _ = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
  )

  print("Loading saved weights and normalization statistics...")
  with open('playground_params_new.pickle', 'rb') as handle:
    params = pickle.load(handle)

  # No longer need to explore, can be deterministic
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  # We prepare to use the viewer
  m, d = get_mj_model_data(env)

  # JIT compile the environment dynamics
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  rng = jax.random.PRNGKey(0)
  print("Compile reset function")
  state = jit_reset(rng)

  step_count = 0
  limiter = rate_limiter.RateLimiter(1 / env._config.ctrl_dt)

  print("Launching MuJoCo passive viewer...")
  with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.user_scn.flags[5] = 1
    viewer.sync()

    while viewer.is_running():
      state = state.replace(data=state.data.replace(
        mocap_pos=jax.numpy.array(d.mocap_pos),
        xfrc_applied=jax.numpy.array(d.xfrc_applied)
      ))

      act_rng, rng = jax.random.split(rng)  # Still need the key
      ctrl, _ = jit_inference_fn(state.obs, act_rng)

      state = jit_step(state, ctrl)
      step_count += 1

      # Push kinematics changes back to the MuJoCo visualization objects
      d.qpos = np.array(state.data.qpos)
      d.qvel = np.array(state.data.qvel)
      d.ctrl = np.array(ctrl)  # Displays muscle activations on the actuator sliders

      # Process forward kinematics for the rendering cpu env
      mujoco.mj_forward(m, d)

      # Stream the reward value to the live visualizer chart (Press F4 in viewer to open)
      d.sensordata[0] = float(state.reward)

      color = np.log(np.maximum(d.ctrl, 0) + 0.0001)
      m.tendon_rgba = (color[:, None] * np.array([0.95, 0.3, 0.3, 1])[None, :]
                       + (1 - color[:, None]) * np.array([0.45, 0.15, 0.15, 1])[None, :])

      viewer.sync()
      limiter.sleep()


def get_mj_model_data(env):
  spec = mujoco.MjSpec.from_file(env.xml_path)
  spec = env.preprocess_spec(spec)

  # Register the user sensor graph with dimensions equal to 1 (reward), visualisation trick
  spec.add_sensor(name="reward", type=mujoco.mjtSensor.mjSENS_USER, dim=1)

  m = spec.compile()
  d = mujoco.MjData(m)
  return m, d


if __name__ == '__main__':
  main()