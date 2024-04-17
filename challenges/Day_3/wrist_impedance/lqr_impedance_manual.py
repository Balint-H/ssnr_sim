# This script is set up so you can control the arm interactively with your mouse.
# After double clicking the target location sphere in the simulation, you can use
# Ctrl + Shift + Right mouse drag to move the target location horizontally.

# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
import scipy
from functools import partial
import time
xml = 'impedance.xml'
dxf = None



def lqr_control(model, data, K, ctrl0, qpos0):
    global dxf
    global armband
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T

    if dxf is None:
        dxf = dx
    else:
        dxf = 0.00*dxf + 1*dx

    if (data.time // model.opt.timestep) % 2000 == 1000:
        data.qpos[-1] = -15
        data.qpos[-2] = 0
        data.qvel[-2] = -2

    if (data.time // model.opt.timestep) % 2000 == 0:
        data.qpos[-2] = -15
        data.qpos[-1] = 0
        data.qvel[-1] = -2

    # LQR control law.
    data.ctrl[:-2] = ctrl0 - K[:-2, :] @ dxf

    # perturbation
    nc = data.ctrl.shape[0] - 2
    data.ctrl[:-2] += perlin(np.atleast_2d(np.ones(nc)*data.time), np.atleast_2d(np.arange(nc))).flatten() / 15
    
    data.ctrl[-2] = data.ctrl[-1]

# This function loads the model and adds arm_control to the simulation loop
def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)
    # return model, data
    if model is not None:
        mujoco.mj_forward(model, data)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        data.qpos[2] -= 0.0005
        qpos0 = data.qpos.copy()  # Save the position setpoint.
        mujoco.mj_inverse(model, data)
        qfrc0 = data.qfrc_inverse.copy()

        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment[:-2, :])
        ctrl0 = ctrl0.flatten()

        nu = model.nu  # Alias for the number of actuators.
        R = np.eye(nu)
        nv = model.nv  # Shortcut for the number of DoFs.

        # Get the Jacobian for the root body (torso) CoM.
        mujoco.mj_resetData(model, data)
        data.qpos = qpos0
        mujoco.mj_forward(model, data)
        jac_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)

        # Get the Jacobian for the left foot.
        jac_lfoot = np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, jac_lfoot, None, model.body('foot_left').id)

        jac_rfoot = np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, jac_rfoot, None, model.body('foot_right').id)

        jac_diff = jac_com# - (jac_lfoot + jac_rfoot)/2
        # jac_diff = jac_diff[:jac_diff.shape[0]-4, :jac_diff.shape[1]-4]
        Qbalance = jac_diff.T @ jac_diff

        # Get all joint names.
        joint_names = [model.joint(i).name for i in range(model.njnt)]

        # Get indices into relevant sets of joints.
        root_dofs = range(6)
        body_dofs = range(6, nv)
        abdomen_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if 'abdomen' in name
               and not 'z' in name
        ]
        left_leg_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if 'left' in name
               and ('hip' in name or 'knee' in name or 'ankle' in name)
               and not 'z' in name
        ]

        right_leg_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if 'right' in name
               and ('hip' in name or 'knee' in name or 'ankle' in name)
               and not 'z' in name
        ]
        balance_dofs = abdomen_dofs + left_leg_dofs + right_leg_dofs
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)

        """We are now ready to construct the Q matrix. Note that the coefficient of the balancing term is quite high. This is due to 3 seperate reasons:
        - It's the thing we care about most. Balancing means keeping the CoM over the foot.
        - We have less control authority over the CoM (relative to body joints).
        - In the balancing context, units of length are "bigger". If the knee bends by 0.1 radians (≈6°), we can probably still recover. If the CoM position is 10cm sideways from the foot position, we are likely on our way to the floor.
        """

        # Cost coefficients.
        BALANCE_COST = 700  # Balancing.
        BALANCE_JOINT_COST = 2  # Joints required for balancing.
        OTHER_JOINT_COST = 0.1  # Other joints.

        # Construct the Qjoint matrix.
        Qjoint = np.eye(nv)
        Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
        Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST
        # Qjoint = Qjoint[:Qjoint.shape[0]-4, :Qjoint.shape[1]-4]
        # Construct the Q matrix for position DoFs.
        Qpos = BALANCE_COST * Qbalance + Qjoint

        # No explicit penalty for velocities.
        Q = np.block([[Qpos, np.zeros((nv, nv))],
                      [np.zeros((nv, 2 * (nv)))]])

        """### Computing the LQR gain matrix $K$

        Before we solve for the LQR controller, we need the $A$ and $B$ matrices. These are computed by MuJoCo's `mjd_transitionFD` function which computes them using efficient finite-difference derivatives, exploiting the configurable computation pipeline to avoid recomputing quantities which haven't changed.
        """

        # Set the initial state and control.
        mujoco.mj_resetData(model, data)
        data.ctrl[:-2] = ctrl0
        data.qpos = qpos0

        # Allocate the A and B matrices, compute them.
        A = np.zeros((2 * nv + model.na, 2 * nv + model.na))
        B = np.zeros((2 * nv + model.na, nu))
        epsilon = 1e-6
        centered = True
        mujoco.mjd_transitionFD(model, data, epsilon, centered, A, B, None, None)
        # A = np.delete(A, [*np.arange(nv-5, nv-1), *np.arange(2*nv-5, 2*nv-1)], axis=0)
        # A = np.delete(A, [*np.arange(nv-5, nv-1), *np.arange(2*nv-5, 2*nv-1)], axis=1)
        # B = np.delete(B, [*np.arange(nv-5, nv-1), *np.arange(2*nv-5, 2*nv-1)], axis=0)
        # # Solve discrete Riccati equation.
        A = A[:-2, :-2]
        B = B[:-2, :]
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute the feedback gain matrix K.
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        data.qpos = qpos0

        mujoco.set_mjcb_control(partial(lqr_control, K=K, qpos0=qpos0, ctrl0=ctrl0))

    return model, data


def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y

def main():
    viewer.launch(loader=load_callback)

if __name__ == '__main__':
    main()
