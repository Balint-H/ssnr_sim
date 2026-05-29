# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
#
# Planar arm control is a classic example to illustrate task-space goals achieved via body kinematics.

import mujoco
import mujoco.viewer as viewer
import os

import numpy as np

from SSNR2026.common_scripts import launch_simulation
from .student_functions import feedback_control


xml = os.path.dirname(__file__) + '/arm_model.xml'


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    # Getting the current position of the interactive target. You can use Ctrl (Cmd on Mac) + Shift + Right click drag
    # to move the target in the horizontal plane.
    joint_target = data.ctrl

    error = joint_target - data.qpos
    velocity_error = np.zeros_like(data.qvel) - data.qvel

    data.qfrc_applied = feedback_control(error, velocity_error)


if __name__ == '__main__':
    launch_simulation(xml, arm_control)

