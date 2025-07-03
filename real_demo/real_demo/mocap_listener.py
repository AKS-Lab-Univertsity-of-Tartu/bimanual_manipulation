import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
import os

# import urx
import time

import mujoco
from mujoco import viewer
import numpy as np

from mj_planner.mjx_planner import cem_planner

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

class MocapListener(Node):
    def __init__(self):
        super().__init__('mocap_listener')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.cem = None
        self.model = None
        self.viewer = None
        self.data = None
        self.xi_mean = None
        self.thetadot = None
        self.target_pos = None
        self.target_rot = None
        self.obstacle_pos = None
        self.obstacle_rot = None
        self.init_position = None
        self.init_rotation = None
        self.init_joint_position = [1.5, -1.8, 1.75, -1.25, -1.6, 0.0]#[-1.52, -1.26, -1.75, -1.69,  1.64, 0.2]

        connected = False
        try:
            # self.rob = urx.Robot("192.168.0.120")
            self.rtde_c = RTDEControl("192.168.0.120")
            self.rtde_r = RTDEReceive("192.168.0.120")
            connected = True
            print("Connection with UR5e established.")
        except:
            print("Could not connect...")

        if connected:
            self.move_to_start()
            self.init_cem()
            self.subscription_object1 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/object1/pose',
                self.object1_callback,
                qos_profile 
            )
            self.subscription_obstacle1 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/obstacle1/pose',
                self.obstacle1_callback,
                qos_profile 
            )
            # self.run_mpc()
            self.timer = self.create_timer(0.05, self.run_mpc)
            # self.close_connection()

    def move_to_start(self):
        # self.rob.movej(self.init_joint_position, acc=0.5, vel=0.5, wait=True)
        self.rtde_c.moveJ(self.init_joint_position, asynchronous=False)
        print("Moved to initial pose.")

    def close_connection(self):
        # self.rob.stopj()
        # self.rob.close()
        self.rtde_c.speedStop()
        self.rtde_c.disconnect()
        print("Disconnected from UR5 Robot")

    def init_cem(self):
        start_time = time.time()
        self.cem =  cem_planner(num_dof=6, num_batch=1000, num_steps=8, maxiter_cem=1,
                           w_pos=5, w_rot=1.5, w_col=10, num_elite=0.05, timestep=0.05)
        print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

        self.model = self.cem.model
        self.data = self.cem.data
        self.data.qpos[:6] = np.array(self.init_joint_position)
        mujoco.mj_forward(self.model, self.data)

        self.xi_mean = np.zeros(self.cem.nvar)
        self.thetadot = np.zeros(6)
        self.target_pos = self.model.body(name="target").pos
        self.target_rot = self.model.body(name="target").quat
        self.obstacle_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='obstacle').id]]
        self.obstacle_rot = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='obstacle').id]]

        self.init_position = self.data.site_xpos[self.model.site(name="tcp").id].copy()
        self.init_rotation = self.data.xquat[self.model.body(name="hande").id].copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        start_time = time.time()
        _ = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                 init_pos=np.array(self.rtde_r.getActualQ()), init_vel=np.zeros(6), 
                                 target_pos=self.target_pos, target_rot=self.target_rot,
                                 obstacle_pos=self.obstacle_pos, obstacle_rot=self.obstacle_rot)
        print(f"Compute CEM: {round(time.time()-start_time, 2)}s")

    def run_mpc(self):

        start_time = time.time()

        # current_pos = np.array(self.rob.getj())
        current_pos = np.array(self.rtde_r.getActualQ())
        # current_vel = self.thetadot
        current_vel = np.array(self.rtde_r.getActualQd())


        self.target_pos = self.model.body(name="target").pos
        self.target_rot = self.model.body(name="target").quat

        self.obstacle_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='obstacle').id]]
        self.obstacle_rot = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='obstacle').id]]

        cost, best_cost_g, best_cost_c, best_vels, best_traj, self.xi_mean = self.cem.compute_cem(xi_mean=self.xi_mean, 
                                init_pos=current_pos, init_vel=current_vel, 
                                target_pos=self.target_pos, target_rot=self.target_rot, 
                                obstacle_pos=self.obstacle_pos, obstacle_rot=self.obstacle_rot)
        
        self.thetadot = np.mean(best_vels[1:5], axis=0)

        # s_time = time.time()
        # self.rob.speedj(self.thetadot, acc=2, min_time=0.2)
        self.rtde_c.speedJ(self.thetadot, acceleration=1.4, time=0.05)
        # print(f'Time: {"%.0f"%((time.time() - s_time)*1000)}ms')

        # self.data.qpos[:6] = self.rob.getj()
        self.data.qpos[:6] = self.rtde_r.getActualQ()

        # self.data.qvel[:6] = thetadot
        mujoco.mj_step(self.model, self.data)

        cost_g = np.linalg.norm(self.data.site_xpos[self.cem.tcp_id] - self.target_pos)   
        self.viewer.sync()
        
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost c: {"%.2f"%(float(best_cost_c))} | Cost: {np.round(cost, 2)}')

    def object1_callback(self, msg):
        pose = msg.pose
        self.model.body(name='target').pos = [-pose.position.x, -pose.position.y, pose.position.z]

    def obstacle1_callback(self, msg):
        pose = msg.pose
        # self.model.body(name='obstacle').pos = [-pose.position.x, -pose.position.y, pose.position.z]
        self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='obstacle').id]] = [-pose.position.x, -pose.position.y, pose.position.z]


def main(args=None):
    rclpy.init(args=args)
    mocap_listener = MocapListener()
    print("Initialized node.")
    rclpy.spin(mocap_listener)

    mocap_listener.close_connection()

    mocap_listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
