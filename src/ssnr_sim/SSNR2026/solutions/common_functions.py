import mujoco
from mujoco import viewer
from functools import partial
import numpy as np


def get_current_and_target_kinematics(model, data, use_traj=False):
  if not use_traj:
    xt, yt, _ = data.mocap_pos[0]
    xvt, yvt = 0, 0
  else:
    xt, yt = desired_hand_position(data.time)
    xvt, yvt = desired_hand_velocity(data.time)
    data.mocap_pos[0][:2] = [xt, yt]
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
  return x, y, xt, yt, xvt, yvt


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

def desired_hand_position(t, T=3):
  t = (t / T) % 2
  reverse = t > 1  # For looping mirrored animation
  t = 2 - t if reverse else t

  g = t * t * t * (6 * t * t - 15 * t + 10)  # For integer exponents this is faster than pow()
  x = -0.1+ 0.11 * g
  y = 0.3 + 0.4 * g
  return np.array([x, y])


# Obtained by differentiating w.r.t. 't'.
def desired_hand_velocity(t, T=3):
  t = (t / T) % 2
  reverse = t > 1
  t = 2 - t if reverse else t
  g_dot = 30 * (t * t + t * t * t * t) - 60 * t * t * t
  x_dot = 0.11 * g_dot
  y_dot = 0.4 * g_dot
  return np.array([x_dot, y_dot]) * (-1 if reverse else 1)


# Transform cartesian position to joint position (possible since there is no redundancy)
def joint_ik_pos(model, cart_pos):
  l_s = model.body("forearm").pos[0]
  l_e = -model.body("hand").pos[1]
  l_e += model.body("wrist_body").pos[0]
  l_e += -model.body("tip").pos[1]
  x, y = cart_pos  # Cartesian position
  qe = np.arccos((x * x + y * y - l_s * l_s - l_e * l_e) / (2 * l_s * l_e))
  alpha = np.arctan2(y, x)
  beta = np.arccos((x * x + y * y + l_s * l_s - l_e * l_e) / (2 * l_s * np.sqrt(x * x + y * y)))
  qs = alpha - beta
  return np.array([qs, qe])


# Transform differential kinematics using Jacobian
def joint_ik_vel(model, cart_vel, q_pos):
  l_s = model.body("forearm").pos[0]
  l_e = -model.body("hand").pos[1]
  l_e += model.body("wrist_body").pos[0]
  l_e += -model.body("tip").pos[1]

  s_s = np.sin(q_pos[0])
  s_se = np.sin(np.sum(q_pos))
  c_s = np.cos(q_pos[0])
  c_se = np.cos(np.sum(q_pos))
  J = np.array([
    [-l_s * s_s - l_e * s_se, -l_e * s_se],
    [l_s * c_s + l_e * c_se, l_e * c_se]
  ])
  q_vel = np.linalg.pinv(J) @ cart_vel
  return q_vel
