# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from etils import epath
from numpy.linalg import pinv, inv
import glfw
from mujoco import mjx

Kp = 100
Kd = 10

Kp_act = 3
Kd_act = 0

xml = (epath.Path(epath.resource_path('mujoco')) / (
        'mjx/test_data/actuator/arm26.xml')).as_posix()


actuators = ['SF', 'SE', 'EF', 'EE', 'BF', 'BE']

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


    rt = np.linalg.norm([xt, yt])
    xt, yt = np.array([xt, yt])/(rt+0.0000001) * np.clip(rt, 0, 1)

    # Current position of arm end in comparison
    x, y, _ = data.body("tip").xpos


    # Jacobian from engine. The jacobian converts differences (e.g. error, velocity) in joint space to differences in
    # task space (Cartesian coords). It depends on the current configuration of the arm, therefore we need to calculate
    # it every frame. We'll use MuJoCo's built in function for getting the matrix.
    J = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacp=J, jacr=None, point=np.array([[x], [y], [0]]), body=model.body("tip").id)
    xvel, yvel, _ = J@data.qvel  # Get task velocity with jacobian
    Ji = pinv(J)  # Invert it so we can go from task space to joint space

    xe, ye = xt-x, yt-y  # Errors in task space
    task_force = Kp * np.array([xe, ye, 0]) - Kd*np.array([xvel, yvel, 0])  # Stiffness and damping in task space!
    f = Ji @ task_force  # desired joint torque
    # Good practice to clip forces to reasonable values
    qfrc_desired = np.clip(f, -100, 100)

    force_error = qfrc_desired - data.qfrc_actuator

    gains = [mujoco.mju_muscleGain(data.actuator(a_name).length[0],
                                   data.actuator(a_name).velocity[0],
                                   model.actuator(a_name).lengthrange.reshape(2, 1),
                                   model.actuator(a_name).acc0[0],
                                   model.actuator(a_name).gainprm[:-1].reshape(9, 1), ) for a_name in actuators]
    gains = np.array(gains)
    moment_matrix = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(moment_matrix,
                            data.actuator_moment,
                            data.moment_rownnz,
                            data.moment_rowadr,
                            data.moment_colind)

    df_dact = gains[None, :] * moment_matrix.T
    activation_error = pinv(df_dact) @ force_error



    # Muscles/cables should only pull
    data.ctrl = np.clip(Kp_act * activation_error - Kd_act * data.act_dot, 0, 1)

def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    spec = mujoco.MjSpec.from_file(filename=xml, assets=None)
    forearm = spec.worldbody.first_body().first_body()
    forearm.name = "forearm"

    tip = forearm.add_body(name="tip",pos=[0.5, 0, 0])
    target = spec.worldbody.add_body(name="target",
                                     mocap=True,
                                     pos=[-0.1, 0.9, 0])
    ball = target.add_geom(name="mocap_geom", size=[0.1, 0, 0], rgba=[0.5, 0.1, 0.1, 0.1], contype=0, conaffinity=0)
    model = spec.compile()

    # `model` contains static information about the modeled system
    model.actuator_dynprm[:, 2] = 0.5
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

