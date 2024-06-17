# This script just loads a MuJoCo xml file without performing any logic in python.
# Alternatively you could just drag and drop an xml file after running "simulate.exe"
# from the MuJoCo directory downloaded.

import mujoco
import mujoco.viewer as viewer
from functools import partial
import pickle
# Change this string to other scenes you may want to load. You can also open the xml in a code editor
# to examine its contents. For more instructions check out the header comments of xml/01_planar_arm.xml
xml = '05_nonlinear_sensing_visu.xml'


def predict(model, data, sv_z, sv_y):
    data.joint("shoulder_z_vis").qpos = sv_z.predict(data.sensordata[None, :])
    data.joint("shoulder_z_vis").qvel= [0]
    data.joint("shoulder_y_vis").qpos = sv_y.predict(data.sensordata[None, :])
    data.joint("shoulder_y_vis").qvel = [0]


# This function is called by the viewer after it is initialized to load the model
def load_callback(model=None, data=None, sv_z=None, sv_y=None):
    mujoco.set_mjcb_control(None)
    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)
    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)
    mujoco.set_mjcb_control(partial(predict, sv_z=sv_z, sv_y=sv_y))
    return model, data


if __name__ == '__main__':
    with open('sv_z.pickle', 'rb') as handle:
        sv_z = pickle.load(handle)
    with open('sv_y.pickle', 'rb') as handle:
        sv_y = pickle.load(handle)
    viewer.launch(loader=partial(load_callback, sv_z=sv_z, sv_y=sv_y))
