import mujoco
from mujoco import viewer
from mujoco import mjx
import os
import numpy as np
import time
import jax

def quaternion_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    return 2 * np.arccos(dot_product)

def rotation_quaternion(angle_deg, axis):
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    w = np.cos(angle_rad / 2)
    x, y, z = axis * np.sin(angle_rad / 2)
    return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))

def quaternion_multiply(q1, q2):
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
		x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
		y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
		z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
		
		return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))


model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.05
data = mujoco.MjData(model)

init_joint_state = [1.5, -1.8, 1.75, -1.25, -1.6, 0]

data.qpos[:6] = init_joint_state
mujoco.mj_step(model, data)
init_position = data.xpos[model.body(name="hande").id]
init_rotation = data.xquat[model.body(name="hande").id]
print(init_rotation)

# mjx_model = mjx.put_model(model)
# mjx_data = mjx.put_data(model, data)

# jax_forward = jax.jit(mjx.forward)
# mjx_data = jax_forward(mjx_model, mjx_data)

file_path = f"{os.path.dirname(__file__)}/data/thetadot.csv" 
thetadot = np.genfromtxt(file_path, delimiter=',')
# thetadot = np.tile(np.zeros(6), (300, 1))


geom_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)]) # [33  7 12 13 18 19 23 27 28 30]

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer_.opt.sitegroup[:] = False  
    viewer_.opt.sitegroup[1] = True 

    start = time.time()
    i = 0
    while viewer_.is_running():
        n+=1
        step_start = time.time()
        # data.qvel[:6] = thetadot[i]

        # qpos = mjx_data.qpos.at[:6].set(data.ctrl[:6])
        # mjx_data = mjx_data.replace(qpos=qpos)
        # mjx_data = jax_forward(mjx_model, mjx_data)

        mujoco.mj_step(model, data)
        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)    

        if i < thetadot.shape[0]-1:
            i+=1
        else:
            data.qvel[:6] = np.zeros(6)
            data.qpos[:6] = init_joint_state
            i=0