import mujoco
from mujoco import viewer
from functools import partial
import numpy as np


def get_current_and_target_position(model, data):
  xt, yt, _ = data.mocap_pos[0]
  data.mocap_pos[0][2] = 0

  # Clipping the target position's distance, otherwise weird behaviour occurs when out of reach
  ls = model.body("forearm").pos[0]
  lw = model.body("wrist_body").pos[0]
  le = -model.body("hand").pos[1]
  lh = -model.body("tip").pos[1]

  rt = np.linalg.norm([xt, yt])
  xt, yt = np.array([xt, yt]) / rt * np.clip(rt, 0, ls + le + lh + lw)
  data.mocap_pos[0][:2] = [xt, yt]

  # Current position of arm end in comparison
  x, y, _ = data.body("tip").xpos
  return x,y, xt, yt


def launch_simulation(xml, arm_control):
  viewer.launch(loader=partial(load_callback, xml, arm_control))


def load_callback(xml, arm_control, model=None, data=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  # `model` contains static information about the modeled system
  model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

  # `data` contains the current dynamic state of the system
  data = mujoco.MjData(model)

  if model is not None:
    # Can set initial state
    data.joint('shoulder').qpos = 0
    data.joint('elbow').qpos = 0

    # The provided "callback" function will be called once per physics time step.
    # (After forward kinematics, before forward dynamics and integration)
    # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
    mujoco.set_mjcb_control(arm_control)

  return model, data