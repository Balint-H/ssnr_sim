# This script just loads a MuJoCo xml file without performing any logic in python.
# Alternatively you could just drag and drop an xml file after running "simulate.exe"
# from the MuJoCo directory downloaded.

import mujoco
import mujoco.viewer as viewer
import matplotlib
matplotlib.use("QtAgg")

# Change this string to other scenes you may want to load. You can also open the xml in a code editor
# to examine its contents. For more instructions check out the header comments of xml/01_planar_arm.xml
xml = r'./welcome.xml'


if __name__ == '__main__':
    viewer.launch(loader=viewer._file_loader(xml), show_left_ui=False, show_right_ui=False)