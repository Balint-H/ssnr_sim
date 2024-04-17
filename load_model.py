# This script just loads a MuJoCo xml file without performing any logic in python.
# Alternatively you could just drag and drop an xml file after running "simulate.exe"
# from the MuJoCo directory downloaded.

import mujoco
import mujoco.viewer as viewer

# Change this string to other scenes you may want to load. You can also open the xml in a code editor
# to examine its contents. For more instructions check out the header comments of xml/01_planar_arm.xml
xml = 'xml/hello_world.xml'


# This function is called by the viewer after it is initialized to load the model
def load_callback(model=None, data=None):
    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)
    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)
    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)
