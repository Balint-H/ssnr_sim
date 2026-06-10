import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
from scipy.signal import filtfilt, butter
import os
from student_functions import *
from common_functions import launch_simulation, get_current_and_target_kinematics, joint_ik_pos, desired_hand_position

Kp = 10
Kd = 0.02

xml = os.path.dirname(__file__) + '/arm_model.xml'

LEARNING_GAIN = 0.1
TRAJECTORY_DURATION = 6.0
DT = 0.005
TOTAL_STEPS = int(TRAJECTORY_DURATION / DT)

ff_torques = np.zeros((TOTAL_STEPS, 2))  # Stores feedforward torques for current iteration
current_fb_torques = np.zeros((TOTAL_STEPS, 2))  # Records feedback torques of the current run
error_history = []  # Records task space errors to calculate MAE

step_counter = 0
iteration_counter = 1


def arm_control(model, data):
  """
  :type model: mujoco.MjModel
  :type data: mujoco.MjData
  """
  global step_counter, iteration_counter, ff_torques, current_fb_torques, error_history

  x, y, xt, yt, xvt, yvt = get_current_and_target_kinematics(model, data, use_traj=True)

  J = np.empty((3, model.nv))
  mujoco.mj_jac(model, data, jacp=J, jacr=None, point=np.array([[x], [y], [0]]), body=model.body("tip").id)

  xvel, yvel, _ = J @ data.qvel
  xe, ye = xt - x, yt - y
  xve, yve = xvt - xvel, yvt - yvel

  position_error = np.array([xe, ye, 0])
  velocity_error = np.array([xve, yve, 0])

  # Track the magnitude of the positioning error for MAE calculation
  error_magnitude = np.sqrt(xe ** 2 + ye ** 2)
  error_history.append(error_magnitude)

  # Compute Feedback Control (FB)
  task_force = Kp*position_error + Kd*velocity_error
  fb_torque = J.T @ task_force  # Feedback joint torques

  # Extract Feedforward Control (FF) for the current timestep
  # If it's the first iteration, this is all zeros.
  current_ff_torque = ff_torques[step_counter]

  # Apply combined control law: Total = FF + FB
  # We restrict the slice to the actual number of joints (model.nv)
  total_torque = current_ff_torque[:model.nv] + fb_torque[:model.nv]
  data.qfrc_applied[:model.nv] = total_torque

  # Record the feedback torque applied at this step for the next iteration
  current_fb_torques[step_counter] = fb_torque

  step_counter += 1
  if step_counter >= TOTAL_STEPS:
    mae = np.mean(error_history)
    print(f"Iteration {iteration_counter} Finished | MAE: {mae:.5f} meters")

    # Update Feedforward for the next iteration: FF_next = FF_current + FB_current
    # This acts as an integrator over iterations
    raw_next_ff = 0.95 * ff_torques + (LEARNING_GAIN * current_fb_torques)
    ff_torques = np.apply_along_axis(lambda x: filtfilt(*butter(2, 10.0 / (0.5 / DT), 'low'), x), 0, raw_next_ff)
    # Reset buffers and tracking metrics
    current_fb_torques = np.zeros((TOTAL_STEPS, 2))
    error_history = []
    step_counter = 0
    iteration_counter += 1

    # Reset MuJoCo simulation to t=0 and initial state
    data.qpos = joint_ik_pos(model, np.array([xt, yt]))
    data.qvel[:] = 0



def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)
    spec = mujoco.MjSpec.from_file(filename=xml, assets=None)

    model = spec.compile()
    data = mujoco.MjData(model)

    if model is not None:
        data.qpos = joint_ik_pos(model, desired_hand_position(0))
        data.qvel[:] = 0
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)
