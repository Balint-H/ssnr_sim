# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw

Kp = 1000
Kd = 100

xml = 'arm_model_tendon.xml'

def weighted_pinv(J, H):
    Hinv = inv(H)
    return Hinv@J.T@pinv(J@Hinv@J.T)


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
    lw = model.body("wrist_body").pos[0]
    le = -model.body("hand").pos[1]
    lh = -model.body("tip").pos[1]

    rt = np.linalg.norm([xt, yt])
    xt, yt = np.array([xt, yt])/rt * np.clip(rt, 0, ls+le+lh+lw)

    # Current position of arm end in comparison
    x, y, _ = data.body("tip").xpos

    # dx/dq
    # Jacobian from engine. The jacobian converts differences (e.g. error, velocity) in joint space to differences in
    # task space (Cartesian coords). It depends on the current configuration of the arm, therefore we need to calculate
    # it every frame. We'll use MuJoCo's built in function for getting the matrix.
    J = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacp=J, jacr=None, point=np.array([[x], [y], [0]]), body=model.body("tip").id)

    H = np.empty((model.nv, model.nv))

    mujoco.mj_fullM(model,H, data.qM)
    H[[(x, x) for x in range(model.nv)]] += 0.01
    H/model.body("upper arm").subtreemass

    xvel, yvel, _ = J@data.qvel  # Get task velocity with jacobian
    Ji = weighted_pinv(J, H)  # Invert it so we can go from task space to joint space

    xe, ye = xt-x, yt-y  # Errors in task space
    task_force = Kp * np.array([xe, ye, 0]) - Kd*np.array([xvel, yvel, 0])  # Stiffness and damping in task space!
    f = J.T @ task_force  # desired joint torque
    # Good practice to clip forces to reasonable values
    qfrc_desired = np.clip(f/10, -10, 10)

    # Convert from joint-space to tendon space
    J_tendon = data.ten_J
    tendon_force = pinv(J_tendon.T) @ qfrc_desired

    # Muscles/cables should only pull
    data.ctrl = np.minimum(0, tendon_force)

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

