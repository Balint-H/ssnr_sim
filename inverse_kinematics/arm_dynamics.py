# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw

Kp = 100
Kd = 0

xml = '../xml/arm_joint_3.xml'


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    # Getting the current position of the interactive target. You can use Ctrl (Cmd on Mac) + Shift + Right click drag
    # to move the target in the horizontal plane.
    xt, yt, _ = data.mocap_pos[0]

    # Clipping the target position's distance, otherwise weird behaviour occurs when out of reach
    ls = model.body("forearm").pos[0]
    le = model.body("hand").pos[0]
    lh = model.body("tip").pos[0]
    rt = np.linalg.norm([xt, yt])
    xt, yt = np.array([xt, yt])/rt * np.clip(rt, 0, ls+le+lh)

    # Current position of arm end in comparison
    x, y, _ = data.body("tip").xpos

    # Jacobian from engine. The jacobian converts differences (e.g. error, velocity) in joint space to differences in
    # task space (Cartesian coords). It depends on the current configuration of the arm, therefore we need to calculate
    # it every frame. We'll use MuJoCo's built in function for getting the matrix.
    J = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacp=J, jacr=None, point=np.array([[x], [y], [0]]), body=model.body("tip").id)
    Ji = pinv(J)  # Invert it so we can go from task space to joint space

    xe, ye = xt-x, yt-y  # Errors in task space
    qe = Ji @ np.array([xe, ye, 0])  # Error vector in joint space

    f = Kp*qe - Kd*data.qvel  # Force based on error + damping to make it stable at high gains.

    # Good practice to clip forces to reasonable values
    data.qfrc_applied = np.clip(f/10, -10, 10)


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

