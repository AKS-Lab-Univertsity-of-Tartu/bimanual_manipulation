import mujoco
from mujoco import viewer
from mujoco import mjx
import os
import numpy as np
import time
import jax
import cv2

def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.rad2deg(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.rad2deg(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.rad2deg(np.arctan2(t3, t4))

    euler = np.round(np.array([X,Y,Z]), 2)

    return euler


model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.02
data = mujoco.MjData(model)

camera = mujoco.MjvCamera() 
camera.lookat[:] = [0.0, 0.0, 0.0]
camera.distance = 3.0  

renderer = mujoco.Renderer(model)
jit_step = jax.jit(mjx.step)

init_joint_state = [1.5, -1.8, 1.75, -1.25, -1.6, 0]
data.qpos[:6] = init_joint_state

mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

file_path = f"{os.path.dirname(__file__)}/data/best_vels.csv" 
thetadot = np.genfromtxt(file_path, delimiter=',')

geom_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)])
# [33  7 12 13 18 19 23 27 28 30]

scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

start = time.time()
i = 0
while True:
    step_start = time.time()
    qvel = mjx_data.qvel.at[:6].set(thetadot[i])
    mjx_data = mjx_data.replace(qvel=qvel)
    mjx_data = jit_step(mjx_model, mjx_data)
    data_temp = mjx.get_data(model, mjx_data)

    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

    # collision_geom = mjx_data.contact.geom
    # collision_dist = mjx_data.contact.dist
    # collision = collision_geom[collision_dist<0]
    # print(collision)

    renderer.update_scene(data_temp, scene_option=scene_option, camera=camera)
    pixels = renderer.render()

    cv2.imshow("img", pixels)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if i < thetadot.shape[0]-1:
        i+=1
    else:
        data.qvel[:6] = np.zeros(6)
        data.qpos[:6] = init_joint_state
        mjx_data = mjx.put_data(model, data)

        i=0
