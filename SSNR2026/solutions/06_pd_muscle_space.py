# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import os

Kp = 10
Kd = 0.3


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
    Ji = weighted_pinv(J, H)  # Invert it so we can go from task space to joint space

    xe, ye = xt-x, yt-y  # Errors in task space
    #xe, ye = xe-xvel*0.1, ye-yvel*0.1
    task_force = Kp * np.array([xe, ye, 0]) - Kd*np.array([xvel, yvel, 0])  # Stiffness and damping in task space!
    f = J.T @ task_force  # desired joint torque
    # Good practice to clip forces to reasonable values
    qfrc_desired = np.clip(f, -1, 1)

    # Convert from joint-space to tendon space
    J_tendon = np.empty((model.ntendon, model.nv))
    mujoco.mju_sparse2dense(J_tendon, data.ten_J, model.ten_J_rownnz, model.ten_J_rowadr, model.ten_J_colind)

    tendon_force = pinv(J_tendon.T) @ qfrc_desired

    # Muscles/cables should only pull
    data.ctrl = -np.minimum(0, tendon_force)


    # We'll visualise the force applied on each tendon:
    color = np.log(data.ctrl+0.0001)
    model.tendon_rgba = (color[:, None] * np.array([0.95, 0.3, 0.3, 1])[None, :]
                         + (1 - color[:, None]) * np.array([0.45, 0.15, 0.15, 1])[None, :])

def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    # We can programmatically edit the XML! Instead of manually changing the file, you can configure it in script.
    # We'll swap the ideal tendon actuators with muscle ones, and add user sensor objects in the scene that we can write
    # data to visualise to.
    spec = mujoco.MjSpec.from_file(filename=xml, assets=None)
    for a in spec.actuators:
        a.set_to_muscle(lmin=0.5, lmax=1.6, vmax=1.5, fpmax=1.3, fvmax=1.2, timeconst=0.01, tausmooth=0,
                        force=-1, scale=200, range=0.75)
        a.gainprm[1] = 1.05
        a.dynprm[1] = 0.04
        a.biasprm[1] = 1.05
        a.ctrlrange = [0, 1]

    model = spec.compile()
    data = mujoco.MjData(model)

    if model is not None:
        data.joint('shoulder').qpos = 0
        data.joint('elbow').qpos =0
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)

