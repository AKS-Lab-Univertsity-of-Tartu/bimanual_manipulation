import mujoco
from mujoco import viewer
import os
from mujoco import mjx
import jax
import numpy as np



model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
data.qpos[:6] = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])

# viewer.launch(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera parameters
    viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
    viewer.cam.distance = 5.0 
    viewer.cam.azimuth = 90.0              # Horizontal angle in degrees
    viewer.cam.elevation = -30.0           # Vertical angle in degrees

    while True:
        mujoco.mj_step(model, data)
        viewer.sync()