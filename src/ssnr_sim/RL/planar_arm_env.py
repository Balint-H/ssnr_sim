from mujoco_playground._src import mjx_env
from ml_collections import config_dict
from mujoco import mjx
import mujoco
import jax
import jax.numpy as jp
from typing import Any

XML_PATH = '../SSNR2026/arm_model_tendon.xml'

default_config = config_dict.create(
  sim_dt=0.004,
  ctrl_dt=0.01,
  episode_length=400,
  reset_pos_range=[0.8, 0.8],
  reset_vel_range=0.8,
  reward_falloff_scale=5,
  reward_excitation_penalty=0.05
)


class PlanarArmTendon(mjx_env.MjxEnv):
  """Planar muscle-tendon arm tracking environment running fully on MJX."""

  def __init__(self, config: config_dict.ConfigDict = default_config):
    super().__init__(config)
    self._mj_spec = mujoco.MjSpec.from_file(filename=XML_PATH, assets=None)
    self._mj_spec = self.preprocess_spec(self._mj_spec)
    self._mj_model = self._mj_spec.compile()

    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
    self._post_init()

  def preprocess_spec(self, spec):
    for a in spec.actuators:
      a.set_to_muscle(lmin=0.5, lmax=1.6, vmax=1.5, fpmax=1.3, fvmax=1.2,
                      timeconst=0.01, tausmooth=0,
                      force=-1, scale=200, range=0.75)
      a.gainprm[1] = 1.05
      a.dynprm[1] = 0.04
      a.biasprm[1] = 1.05
      a.ctrlrange = [0, 1]
    return spec

  def _post_init(self) -> None:
    self._shoulder_qposadr = self._mj_model.joint("shoulder").id
    self._elbow_qposadr = self._mj_model.joint("elbow").id
    self._wrist_qposadr = self._mj_model.joint("wrist").id
    self._tip_body_id = self._mj_model.body("tip").id

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_q, rng_v, rng_target1, rng_target2 = jax.random.split(rng, 5)

    low_bounds = self._mj_model.jnt_range[:, 0] * self._config.reset_pos_range[0]
    high_bounds = self._mj_model.jnt_range[:, 1] * self._config.reset_pos_range[1]
    v_bound = ((self._mj_model.jnt_range[:, 1] - self._mj_model.jnt_range[:, 0])
               * self._config.reset_vel_range)
    new_qpos = jax.random.uniform(rng_q, shape=self._mj_model.qpos0.shape,
                                  minval=low_bounds,
                                  maxval=high_bounds)
    new_qvel = jax.random.uniform(rng_v, shape=self._mj_model.qpos0.shape,
                                  minval=-v_bound,
                                  maxval=v_bound)

    data = mjx.make_data(
      self.mj_model,
      impl=self._mjx_model.impl.value,
    )
    data = data.replace(qpos=new_qpos, qvel=new_qvel, )
    data = mjx.forward(self.mjx_model, data)
    data = data.replace(act=jp.ones_like(data.act) * jp.nan)

    radius = jax.random.uniform(rng_target1, (), minval=0.2, maxval=0.5)
    angle = jax.random.uniform(rng_target2, (), minval=-jp.pi / 6, maxval=jp.pi / 1.5)
    target_pos = jp.array([radius * jp.cos(angle), radius * jp.sin(angle)])
    data = data.replace(mocap_pos=jp.array([[target_pos[0], target_pos[1], 0.]]))
    metrics = {
      "reward/tracking": jp.zeros(()),
      "reward/excitation_penalty": jp.zeros(()),
      "distance": jp.zeros(())
    }

    info = {"rng": rng, "target": target_pos}
    obs = self._get_obs(data, info)
    reward_val, done = jp.zeros(2)

    return mjx_env.State(data, obs, reward_val, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    ctrl = jp.clip(action, 0.0, 1.0)
    data = state.data
    data = data.replace(act=jp.where(jp.isnan(data.act), ctrl, data.act))
    data = data.replace(mocap_pos=jp.array([[data.mocap_pos[0][0], data.mocap_pos[0][1], 0.]]))

    data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

    reward_val, metrics = self._get_reward(data, ctrl, state.info, state.metrics)
    obs = self._get_obs(data, state.info)

    done = self._get_done(data)

    return state.replace(data=data, obs=obs, reward=reward_val, done=done, metrics=metrics)

  def _get_done(self, data):
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    at_low = data.qpos <= self._mjx_model.jnt_range[:, 0]
    at_high = data.qpos >= self._mjx_model.jnt_range[:, 1]
    done = done | jp.any(at_low | at_high)
    done = done.astype(float)
    return done

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    tip_pos = data.xpos[self._tip_body_id][:2]
    target_pos = data.mocap_pos[0][:2]

    return jp.concatenate([
      jp.sin(data.qpos),
      jp.cos(data.qpos),
      data.qvel,
      tip_pos,
      target_pos,
      target_pos - tip_pos
    ])

  def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any], metrics: dict[str, Any]):
    tip_pos = data.xpos[self._tip_body_id][:2]
    target_pos = info["target"]

    dist = jp.linalg.norm(tip_pos - target_pos)
    metrics["distance"] = dist

    tracking = jp.exp(-self._config.reward_falloff_scale * dist)
    metrics["reward/tracking"] = tracking

    excitation_penalty = -(self._config.reward_excitation_penalty
                           * jp.sum(jp.square(action)))
    metrics["reward/excitation_penalty"] = excitation_penalty

    return tracking + excitation_penalty, metrics

  @property
  def xml_path(self) -> str:
    return XML_PATH

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
