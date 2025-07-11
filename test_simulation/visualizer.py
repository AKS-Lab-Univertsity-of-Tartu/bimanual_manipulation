import os
import numpy as np
import time


import mujoco
from mujoco import viewer


class Visualizer():
    def __init__(self, ctrl: bool=False):
        self.init_joint_state = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, 0, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])

        if ctrl:
            model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene_control.xml" 
        else:
            model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:len(self.init_joint_state)] = self.init_joint_state
        self.data.ctrl[:6] = self.init_joint_state[:6]

        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    for i in range(self.model.njnt)]

        print("Joint names:")
        for name in joint_names:
            print(name)

        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                        for i in range(self.model.nu)]

        print("Control inputs (actuators):")
        for name in actuator_names:
            print(name)


        


    # def view_model(self):
    #     viewer.launch(self.model, self.data)

    def view_model(self):
        with viewer.launch_passive(self.model, self.data) as viewer_:
            viewer_.cam.distance = 4
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer_.opt.sitegroup[:] = False  
            viewer_.opt.sitegroup[1] = True 

            while viewer_.is_running():
                step_start = time.time()
                # self.data.qpos[:12] = self.data.ctrl[:12]

                mujoco.mj_step(self.model, self.data)
                viewer_.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)




def main():
    viz = Visualizer(ctrl=False)
    viz.view_model()


if __name__=="__main__":
    main()