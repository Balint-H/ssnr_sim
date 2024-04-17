"""
This is one way to solve the challenge using a rule based controller. Try your own solution before having a look here!
"""
import mujoco
import mujoco.viewer as viewer
from functools import partial
from utility.generalized_coords import get_generalized_coordinate_dict, get_pos_error, \
    get_vel_error, smooth_loop_dict
from utility.realtime_plotting import DataCollecter, subsampled_execution
import numpy as np


# Change this string to other scenes you may want to load. You can also open the xml in a code editor
# to examine its contents. For more instructions check out the header comments of xml/01_planar_arm.xml
xml = 'xml/pd_track.xml'

data_collecter = DataCollecter(20, 3, plot=False)
data_plotter = DataCollecter(300, 1, min_max_scales=[0, 1])


def ankle_control(model, data, precalc_mocap_dictionary, mocap_metadata):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    :param precalc_mocap_dictionary: A dictionary with segment name as key, and a tuple of positions and rotations as
                                     values.
    :param mocap_metada: A dictionary containing the time between frames (frame_time) and duration of mocap
                         (frame_count)
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    # this method applies the necessary forces to mimic the walk cycle, no need to edit it.
    track_mocap(model, data, precalc_mocap_dictionary, mocap_metadata)

    # data.ctrl holds the values of the sliders from the UI. You can also write values to data.ctrl, and it
    # will be applied to the simulation
    ctrls = data.ctrl
    _, param_1, param_2, param_3 = ctrls
    pass

    # if np.median(data_collecter.data_deque, axis=0)[1] > param_1 * 5:
    #     data.ctrl[0] = 0
    if np.median(data_collecter.data_deque, axis=0)[1] > param_1 * 10:
        data.ctrl[0] = param_3
    if np.median(data_collecter.data_deque, axis=0)[1] < param_2 * -10:
        data.ctrl[0] = param_3/10

    subsampled_execution(lambda: data_collecter.add_data(data.sensor("ltibiaACC").data), data, 3)

    smooth_reward(data, model)
    subsampled_execution(lambda: data_plotter.add_data([data.userdata[0]]), data, 3)


# You do not need to edit this method, but feel free to experiment with adjusting values
def track_mocap(model, data, precalc_mocap_dictionary, mocap_metadata,):
    """
    This method applies simple PD tracking to all degrees of freedom to follow a prerecorded walk cycle.
    PD tracking allows deviation from the motion, which makes it suitable for modulating it with the orthosis.

    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    :param precalc_mocap_dictionary: A dictionary containing a dictionary of joint positions, and another for velocities
    :param mocap_metadata: A dictionary of the time between frames in the dictionary, and the duration of a gait cycle.
    """

    # Convert the current simulation time to the index of the corresponding frame in our reference motion
    t_idx = int(data.time // mocap_metadata["frame_time"])

    # Use the modulo operator to loop back to the start once a gait cycle is finished
    t_idx = t_idx % mocap_metadata["frame_count"]

    # PD parameters for following the refernce motion
    stiffness = 500
    damping = 3
    for name in precalc_mocap_dictionary["pos"].keys():
        if name == "freejoint":
            continue
        pos_error = get_pos_error(data, precalc_mocap_dictionary["pos"], t_idx, name)
        vel_error = get_vel_error(data, precalc_mocap_dictionary["vel"], t_idx, name)

        # qfrc_bias holds the (-1*) gravity and coriolis forces, adding it counteracts their effect.
        # This basically represents a tonic motor system, a base motor output on top of which additional modulation
        # is applied.
        gravity_compensation = 0.0 * data.joint(name).qfrc_bias

        data.joint(name).qfrc_applied = stiffness * pos_error + damping * vel_error + gravity_compensation

    # Modelling no active actuation of the ankle.
    # However, a passive stiffness is applied to it in the model description file (the xml),
    # bringing it to plantarflexion.
    data.joint("lfootrx").qfrc_applied[0] *= 0.00

    # Setting the root kinematics. This is a bit of a cheat, but necessary for maintaining balance
    # in lieu of a proper locomotion policy that can recover from trips. Note that we also need
    # to set higher order kinematics, otherwise the model will accumulate momentum and start to drift
    data.joint("freejoint").qpos = precalc_mocap_dictionary["pos"]["freejoint"][:, t_idx]
    data.joint("freejoint").qvel = 0
    data.joint("freejoint").qacc = 0

    # We'll fake forward motion by moving the ground instead. Hurrah for Galilean relativity!
    mocap_body_idx = int(model.body("floor").mocapid)  # mocap bodies are only moved by us, unaffected by physics
    data.mocap_pos[mocap_body_idx][1] = - (1.5*data.time)
    return data


# This function loads the environment and the reference animation for the gait cycle
def load_callback(model=None, data=None):
    # `model` contains static information about the modeled system
    # (e.g. number of DoFs, inertia, contact characteristics like friction etc.)
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)
    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)
    if model is not None:
        # Load generalized joint position information.
        target_pos = get_generalized_coordinate_dict("data/qpos.csv")
        target_pos["freejoint"][:2, :] = 0  # Only move the root body up and down
        del target_pos["lfootrz"]  # The model we'll control has only 1 DoF ankle, so we'll remove the extra data
        # We'll cut the data to a single gait cycle,
        # and make sure it smoothly loops using polynomial convolutional filters
        target_pos = smooth_loop_dict(target_pos, slice(100, 306))

        target_vel = get_generalized_coordinate_dict("data/qvel.csv")
        # Need to set velocities as second order kinematics will influence position updates
        target_vel["freejoint"][:2, :] = 0
        del target_vel["lfootrz"]
        target_vel = smooth_loop_dict(target_vel, slice(100, 306))

        # We'll need this data to find the current frame in our animation
        mocap_metadata = {"frame_time": 0.005, "frame_count": 206}
        precalc_mocap_dictionary = {"pos": target_pos, "vel": target_vel}

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(partial(ankle_control,
                                        precalc_mocap_dictionary=precalc_mocap_dictionary,
                                        mocap_metadata=mocap_metadata))
    return model, data


def smooth_reward(data, model, smoothing_factor=0.998):
    data.userdata[0] = (1-smoothing_factor) * reward(data, model) + smoothing_factor * data.userdata[0]
    return data.userdata[0]


def reward(data, model):
    height_reward = np.clip(np.exp(0.1 * data.joint("lfootrx").qpos), 0, 1)
    trip_force_reward = (np.exp(-100 * data.sensor("ltoeForce").data[0])-0.6) * 1/0.4
    actuation_reward = np.exp(-1 * data.ctrl[0])
    return height_reward * trip_force_reward * actuation_reward


if __name__ == '__main__':
    viewer.launch(loader=load_callback)
