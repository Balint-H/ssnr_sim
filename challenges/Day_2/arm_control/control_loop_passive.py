import time
import mujoco
import mujoco.viewer


def main():
    m = mujoco.MjModel.from_xml_path(r'C:\Users\hbkm9\Documents\Projects\Source_files\Python\ssnr23_ws4\challenges\Day_3\arm_design\solution_scene_wall.xml')
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step1(m, d)

            # Add your control logic here
            time.sleep(0.0001)

            mujoco.mj_step2(m, d)


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                print(time_until_next_step)
                time.sleep(time_until_next_step)
    pass

if __name__ == '__main__':
    main()