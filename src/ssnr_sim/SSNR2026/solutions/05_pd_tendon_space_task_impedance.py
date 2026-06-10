# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import os

Kp = 500
Kd = 3
virtual_mass = 1


xml = os.path.dirname(__file__) + '/arm_model_tendon.xml'


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
    data.mocap_pos[0][2] = 0

    # Clipping the target position's distance, otherwise weird behaviour occurs when out of reach
    ls = model.body("forearm").pos[0]
    lw = model.body("wrist_body").pos[0]
    le = -model.body("hand").pos[1]
    lh = -model.body("tip").pos[1]

    rt = np.linalg.norm([xt, yt])
    xt, yt = np.array([xt, yt])/rt * np.clip(rt, 0, ls+le+lh+lw)
    data.mocap_pos[0][:2] = [xt, yt]

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
    H = H/model.body("upper arm").subtreemass

    xvel, yvel, _ = J@data.qvel  # Get task velocity with jacobian
    Lambda = pinv(J@inv(H)@J.T)  # Operational mass matrix

    xe, ye = xt-x, yt-y  # Errors in task space
    task_force = Kp * np.array([xe, ye, 0]) - Kd*np.array([xvel, yvel, 0])  # Stiffness and damping in task space!
    f = J.T @ Lambda @task_force/virtual_mass  # desired joint torque
    # Good practice to clip forces to reasonable values
    qfrc_desired = np.clip(f, -100, 100)

    # Convert from joint-space to tendon space
    J_tendon = np.empty((model.ntendon, model.nv))
    mujoco.mju_sparse2dense(J_tendon, data.ten_J, model.ten_J_rownnz, model.ten_J_rowadr, model.ten_J_colind)

    tendon_force = pinv(J_tendon.T) @ qfrc_desired

    # Muscles/cables should only pull
    data.ctrl = tendon_force

    # We'll visualise the force applied on each tendon:
    color = 1/(1 + np.exp(-data.ctrl))
    model.tendon_rgba = (color[:, None] * np.array([0.95, 0.3, 0.3, 1])[None, :]
                         + (1 - color[:, None]) * np.array([0.3, 0.3, 0.95, 1])[None, :])

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

