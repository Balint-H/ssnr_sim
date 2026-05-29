# If you have a given position in task space you want to reach

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import os
from .student_functions import *
from .common_scripts import launch_simulation, get_current_and_target_position

Kp = 100
Kd = 0

xml = os.path.dirname(__file__) + '/arm_model.xml'


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    # Check out the definition of this function that reads the simulation state, available in the file common_scripts.py
    x, y, xt, yt = get_current_and_target_position(model, data)

    # dx/dq
    # Jacobian from engine. The jacobian converts differences (e.g. error, velocity) in joint space to differences in
    # task space (Cartesian coords). It depends on the current configuration of the arm, therefore we need to calculate
    # it every frame. We'll use MuJoCo's built in function for getting the matrix.
    J = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacp=J, jacr=None, point=np.array([[x], [y], [0]]), body=model.body("tip").id)

    xvel, yvel, _ = J@data.qvel  # Get task velocity with jacobian

    xe, ye = xt-x, yt-y  # Errors in task space
    position_error = np.array([xe, ye, 0])
    velocity_error = -np.array([xvel, yvel, 0])
    task_force = feedback_control(position_error, velocity_error)
    qfrc_desired = J.T @ task_force  # desired joint torque
    # Convert from joint-space to tendon space
    J_tendon = np.empty((model.ntendon, model.nv))
    mujoco.mju_sparse2dense(J_tendon, data.ten_J, model.ten_J_rownnz, model.ten_J_rowadr, model.ten_J_colind)

    tendon_force = pinv(J_tendon.T) @ qfrc_desired

    # Muscles/cables should only pull, experiment with disabling pushing: tendon_force = np.minimum(tendon_force, 0)
    data.ctrl = tendon_force

    # We'll visualise the force applied on each tendon:
    color = 1/(1 + np.exp(-2*data.ctrl))
    model.tendon_rgba = (color[:, None] * np.array([0.3, 0.3, 0.95, 1])[None, :]
                         + (1 - color[:, None]) * np.array([0.95, 0.3, 0.3, 1])[None, :])


if __name__ == '__main__':
    launch_simulation(xml, arm_control)

