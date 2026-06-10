import dm_control.mjcf as mjcf
import mujoco
from mujoco import viewer

def main():
    rig = mjcf.from_path(r"scene_rig.xml")
    rig_attach = rig.find("site", "attach_site")

    arm = mjcf.from_path(r"elbow.xml")
    arm_attach = arm.find("site", "attach_site")

    hand = mjcf.from_path(r"hand.xml")

    arm_attach.attach(hand)
    rig_attach.attach(arm)

    model = mujoco.MjModel.from_xml_string(rig.to_xml_string(), rig.get_assets())
    viewer.launch(model)

    pass

if __name__ == '__main__':
    main()