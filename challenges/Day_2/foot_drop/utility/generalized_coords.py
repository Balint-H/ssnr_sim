import numpy as np
from itertools import groupby
import mujoco
import scipy.signal as sig


def get_generalized_coordinate_dict(filepath):
    """
    Loading function for our csv files containing the joint coordinates of our gait cycle. It's specific
    for our idiosyncratic files, made only for this programming challenge.
    """
    data = np.genfromtxt(filepath, dtype=float, delimiter=',', names=True)
    grouped_names = list([(k, list(g)) for k, g in groupby(data.dtype.names, lambda n: n.split("_")[0])])
    grouped_data = {key: np.array([data[name] for name in group]) for key, group in grouped_names}
    return grouped_data


def smooth_for_continuous_loop(arr):
    """
    We'll copy the array to its start and end before smoothing, so it interpolates the slight discontinuity
    when looping back to the start, and to remove boundary effects of the convolutional filter
    """
    extended_arr = np.concatenate([arr, arr, arr], axis=1)  # Assuming time indexed second
    smoothed = sig.savgol_filter(extended_arr, 71, 3, axis=1)  # Hardcoded value, could be parametrized
    return smoothed[:, arr.shape[1]:arr.shape[1] * 2]  # Crop to original length


def smooth_loop_dict(coord_dict, slice_in):
    """
    This method applies the looping smoothing for each item in a dictionary
    """
    for k, v in coord_dict.items():
        smoothed = smooth_for_continuous_loop(v[:, slice_in])  # Assuming time sliced at second index
        coord_dict[k] = smoothed
    return coord_dict


def get_pos_error(data, mocap_dict, t_idx, name):
    """
    The main motivation for a separate method for this is getting quaternion differences; rotations are 4D
    (to avoid singularities), but their difference is 3D. Hence we need special cases for free and ball joints.
    Quaternion differences are basically axis-angle like representation.
    """
    if mocap_dict[name].shape[0] == 7:
        rot_diff = np.empty(3)
        mujoco.mju_subQuat(rot_diff, mocap_dict[name][3:, t_idx], data.joint(name).qpos[3:])
        pos_diff = mocap_dict[name][:3, t_idx] - data.joint(name).qpos[:3]
        return np.concatenate([pos_diff, rot_diff])

    if mocap_dict[name].shape[0] == 4:
        res = np.empty(3)
        mujoco.mju_subQuat(res, mocap_dict[name][:, t_idx], data.joint(name).qpos)
        return res
    else:
        return mocap_dict[name][:, t_idx] - data.joint(name).qpos


def get_vel_error(data, mocap_dict, t_idx, name):
    """
    For consistency with pos error
    """
    return mocap_dict[name][:, t_idx] - data.joint(name).qvel