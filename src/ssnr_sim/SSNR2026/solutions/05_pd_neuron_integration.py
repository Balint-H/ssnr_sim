import os
import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
from torch.ao.nn.intrinsic import qat

from common_functions import launch_simulation, get_current_and_target_kinematics

from neuron_functions import (
    build_mu_mapping,
    init_pool_state,
    force_to_current,
    detect_spike_refractory_pool,
    update_excitation,
    DEFAULT_LIF,
)



Kp = 300
Kd = 5


N_PER_MUSCLE = 5        # motoneurons per muscle
F_MAX       = 6.0          
I_MIN       = 1.0e-10    
DRIVE_SCALE = 0.1
I_MAX       = 7e-10
TAU_EXC     = 20e-3


assignment  = None       
mu_mapping  = None       
v_state     = None       
last_spike  = None       
E_state     = None       


xml = os.path.dirname(__file__) + '/arm_model_tendon.xml'


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    global v_state, last_spike, E_state
# `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    x, y, xt, yt, xvt, yvt = get_current_and_target_kinematics(model, data)

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
    xacc, yacc, _ = J@data.qacc_smooth

    xe, ye = xt - (x + xvel * 0.07), yt - (y + yvel * 0.07)  # Errors in task space, with forecasting
    xve, yve = xvt - (xvel + xacc*0), yvt - (yvel + yacc*0)
    position_error = np.array([xe, ye, 0])
    velocity_error = np.array([xve, yve, 0])
    task_force = Kp * position_error + Kd * velocity_error
    f = J.T @ task_force
    qfrc_desired = np.clip(f, -1, 1)

    J_tendon = np.empty((model.ntendon, model.nv))
    mujoco.mju_sparse2dense(J_tendon, data.ten_J, model.ten_J_rownnz,
                            model.ten_J_rowadr, model.ten_J_colind)
    tendon_force = pinv(J_tendon.T) @ qfrc_desired

   
    F_target = -np.minimum(0, tendon_force)            

    
    dt    = model.opt.timestep
    t_now = data.time

        

    I_per_muscle = force_to_current(F_target, F_max=F_MAX, I_min=I_MIN, I_max=I_MAX)
    I_per_neuron = assignment @ I_per_muscle            

    spikes, v_state, last_spike = detect_spike_refractory_pool(
        v_state, last_spike, t_now, dt, I_per_neuron, params=DEFAULT_LIF
    )


    E_state = update_excitation(spikes, E_state, mu_mapping, dt, tau_exc=TAU_EXC)

    data.ctrl[:] = np.clip(E_state, 0.0, 1.0)   

   
    color = np.log(data.ctrl + 1e-4)
    model.tendon_rgba = (color[:, None] * np.array([0.95, 0.3, 0.3, 1])[None, :]
                         + (1 - color[:, None]) * np.array([0.45, 0.15, 0.15, 1])[None, :])


def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    spec = mujoco.MjSpec.from_file(filename=xml, assets=None)
    for a in spec.actuators:
        a.set_to_muscle(lmin=0.5, lmax=1.6, vmax=1.5, fpmax=1.3, fvmax=1.2,
                        timeconst=0.01, tausmooth=0, force=-1, scale=200, range=0.75)
        a.gainprm[1] = 1.05
        a.dynprm[1]  = 0.04
        a.biasprm[1] = 1.05
        a.ctrlrange  = [0, 1]

    model = spec.compile()
    data  = mujoco.MjData(model)

    global assignment, mu_mapping, v_state, last_spike, E_state

    n_muscles = model.nu
    n_neurons = n_muscles * N_PER_MUSCLE

    assignment, mu_mapping = build_mu_mapping(n_muscles, N_PER_MUSCLE,
                                              drive_scale=DRIVE_SCALE, seed=11)
    v_state, last_spike, E_state = init_pool_state(n_neurons, n_muscles,
                                                   el=DEFAULT_LIF["el"])

    print(f"[pool] {n_muscles} muscles x {N_PER_MUSCLE} MN = {n_neurons} neurons")
    print(f"[pool] dt = {model.opt.timestep*1000:.2f} ms, tau_exc = {TAU_EXC*1000:.1f} ms")
    print(f"[pool] F_MAX = {F_MAX}, I_MAX = {I_MAX:.2e} A")

    if model is not None:
        data.joint('shoulder').qpos = 0
        data.joint('elbow').qpos    = 0
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)