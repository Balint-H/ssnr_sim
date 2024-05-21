# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw

Kp = 100
Kd = 1

xml = 'arm_model.xml'


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
    joint_velocity = data.qvel

    data.qfrc_applied = error*Kp + Kd*(0-joint_velocity)


def load_callback(model=None, data=None):
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
        data.joint('elbow').qpos =0

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)

