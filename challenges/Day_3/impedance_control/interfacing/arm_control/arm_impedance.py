# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
from interfacing.arm_control.realtime_plotting import DataCollecter, subsampled_execution

from interfacing.parallel_armband import ParallelSerialArmbandManager

xml = 'scene_wall.xml'
regressor_path = ""  # Set the path to the csv file containing the mixing weigths for linear regression


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """

    # Collecting EMG data
    cur_activation = armband.get_data()
    if len(cur_activation) > 0:
        emg = np.array(cur_activation[0], dtype=float).reshape((12, -1))
        new_rms = np.sqrt(np.mean(np.array(emg[:], float) * np.array(emg[:], float), axis=1))
        data.userdata[12:] = new_rms

    data.userdata[:12] = 0.9 * data.userdata[:12] + 0.1 * data.userdata[12:]

    flexion_rms = np.mean(data.userdata[flexion_idx])
    extension_rms = np.mean(data.userdata[extension_idx])

    if flexion_rms != 0:
        data.actuator("flexion_rms").ctrl = flexion_rms
        data.actuator("extension_rms").ctrl = extension_rms

    # Getting kinematic reference targets
    if not np.all(linear_weights == 0):
        data.actuator("q_target").ctrl = linear_weights @ data.userdata

    q_target = data.actuator("q_target").ctrl
    q_err = q_target - data.joint("flexion").qpos
    q_vel = data.joint("flexion").qvel

    a1, a2, a3, a4 = [data.actuator(param).ctrl for param in ["a1", "a2", "a3", "a4"]]

    STI = a1 * data.actuator("flexion_rms").ctrl/50 + (1 - a1) * data.actuator("extension_rms").ctrl/50
    Kp = a2 * STI + a3
    Kv = np.sqrt(Kp/a4)

    data.joint("flexion").qfrc_applied = np.clip(Kp * q_err - Kv * q_vel, -150, 150)
    if GRAVITY_COMP:
        data.joint("flexion").qfrc_applied += data.joint("flexion").qfrc_bias

    subsampled_execution(lambda: emg_plotter.add_data([flexion_rms, extension_rms]), data, 4)



    pass


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    data.actuator("a4").ctrl = 1
    if model is not None:

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data

def main():
    global armband
    armband.start()
    time.sleep(2)
    viewer.launch(loader=load_callback)


armband = ParallelSerialArmbandManager(port="COM3", byte_count=960)
emg_plotter = DataCollecter(nchannels=2, min_max_scales=[0, 1])

FLEXION_START = 6
ELECTRODE_COUNT = 12

GRAVITY_COMP = False

# We'll use the modulo operator to wrap around, assuming that electrode numbers increase monotonically
flexion_idx = [idx % ELECTRODE_COUNT for idx in range(FLEXION_START, FLEXION_START+ELECTRODE_COUNT//2) ]
extension_idx =[idx % ELECTRODE_COUNT for idx in range(FLEXION_START+ELECTRODE_COUNT//2, FLEXION_START+ELECTRODE_COUNT) ]

flexion_idx = flexion_idx[2:5]
extension_idx = extension_idx[2:5]

linear_weights = np.zeros(12)
if regressor_path:
    linear_weights = np.genfromtxt('my_file.csv', delimiter=',')

if __name__ == '__main__':
    main()

