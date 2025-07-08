import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
import json
import csv
import time

import mujoco
from mujoco import viewer
import numpy as np

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')

USE_HARDWARE = False

RECORD_DATA = False
PLAYBACK = True
idx = "1".zfill(3)


class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        self.init_joint_state = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
        self.trajectory = list()

        model_path = os.path.join(PACKAGE_DIR, 'ur5e_hande_mjx', 'scene.xml')

        # folder = 'manual'
        folder = 'planner'

        self.pathes = {
            "setup": os.path.join(PACKAGE_DIR, 'data', folder, 'setup', f'setup_{idx}.csv'),
            "trajectory": os.path.join(PACKAGE_DIR, 'data', folder, 'trajectory', f'trajectory_{idx}.csv'),
        }
        if RECORD_DATA:
            self.data_files = dict()
            self.data_files['setup'] = csv.DictWriter(open(self.pathes['setup'], "w+"), fieldnames=['table_1', 'marker_1', 'table_2', 'marker_2'])
            self.data_files['trajectory'] = csv.DictWriter(open(self.pathes['trajectory'], "w+"), fieldnames=['timestamp', 'theta', 'thetadot', 'target_1', 'target_2'])
            self.data_files['setup'].writeheader()
            self.data_files['trajectory'].writeheader()

        elif PLAYBACK:
            self.data_files = dict()
            for key, value in self.pathes.items():
                self.data_files[key] = list(csv.DictReader(open(value, "r")))


        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.05
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:12] = self.init_joint_state

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        if not PLAYBACK:

            if USE_HARDWARE:
                self.rtde_c_1 = RTDEControl("192.168.0.120")
                self.rtde_r_1 = RTDEReceive("192.168.0.120")

                self.rtde_c_2 = RTDEControl("192.168.0.124")
                self.rtde_r_2 = RTDEReceive("192.168.0.124")

            self.subscription_table1 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/table1/pose',
                self.table1_callback,
                qos_profile 
            )

            self.subscription_table2 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/table2/pose',
                self.table2_callback,
                qos_profile 
            )

            self.subscription_object1 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/object1/pose',
                self.object1_callback,
                qos_profile 
            )

            self.subscription_object2 = self.create_subscription(
                PoseStamped,
                '/vrpn_mocap/object2/pose',
                self.object2_callback,
                qos_profile 
            )

            self.timer = self.create_timer(0.05, self.view_model)
        else:
            self.setup = self.data_files['setup'][0]

            self.model.body(name='table_1').pos = json.loads(self.setup['table_1'])
            self.model.body(name='table1_marker').pos = json.loads(self.setup['marker_1'])
            self.model.body(name='table_2').pos = json.loads(self.setup['table_2'])
            self.model.body(name='table2_marker').pos = json.loads(self.setup['marker_2'])

            self.step_idx = 0
            self.timer = self.create_timer(0.05, self.view_playback)

    def view_model(self):
        step_start = time.time()

        if USE_HARDWARE:
            theta_1 = self.rtde_r_1.getActualQ()
            theta_2 = self.rtde_r_2.getActualQ()
            theta = np.concatenate((theta_1, theta_2), axis=None)

            thetadot_1 = self.rtde_r_1.getActualQd()
            thetadot_2 = self.rtde_r_2.getActualQd()
            thetadot = np.concatenate((thetadot_1, thetadot_2), axis=None)
        else:
            theta = self.data.qpos[:12]
            thetadot = self.data.qvel[:12]

        self.data.qpos[:12] = theta

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        if RECORD_DATA:
            self.record_data(theta=theta, thetadot=thetadot)

        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)   

    def view_playback(self):
        step_start = time.time()

        theta = json.loads(self.data_files['trajectory'][self.step_idx]['theta'])

        target_1 = json.loads(self.data_files['trajectory'][self.step_idx]['target_1'])
        target_2 = json.loads(self.data_files['trajectory'][self.step_idx]['target_2'])

        self.model.body(name='target_1').pos = target_1[:3]
        self.model.body(name='target_1').quat = target_1[3:]

        self.model.body(name='target_2').pos = target_2[:3]
        self.model.body(name='target_2').quat = target_1[3:]

        self.data.qpos[:12] = theta

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        if self.step_idx < len(self.data_files['trajectory'])-1:
            self.step_idx += 1
        else:
            self.step_idx = 0

        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step) 

    def table1_callback(self, msg):
        marker_pose =  [-msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z]
        marker_diff = marker_pose-self.model.body(name='table1_marker').pos
        table1_pose = self.model.body(name='table_1').pos + marker_diff
        self.model.body(name='table_1').pos = table1_pose
        self.model.body(name='table1_marker').pos = marker_pose

    def table2_callback(self, msg):
        marker_pose =  [-msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z]
        marker_diff = marker_pose-self.model.body(name='table2_marker').pos
        table2_pose = self.model.body(name='table_2').pos + marker_diff
        self.model.body(name='table_2').pos = table2_pose
        self.model.body(name='table2_marker').pos = marker_pose

    def object1_callback(self, msg):
        marker_pose =  [-msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z]
        self.model.body(name='target_1').pos = marker_pose

    def object2_callback(self, msg):
        marker_pose =  [-msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z]
        self.model.body(name='target_2').pos = marker_pose

    def close_connection(self):
        self.rtde_c_1.speedStop()
        self.rtde_c_1.disconnect()
        self.rtde_c_2.speedStop()
        self.rtde_c_2.disconnect()
        print("Disconnected from UR5 Robot")

    def record_data(self, theta, thetadot):
        timestamp = time.time()

        setup = {
            "table_1": str(self.model.body(name='table_1').pos.tolist()),
            "marker_1": str(self.model.body(name='table1_marker').pos.tolist()),
            "table_2": str(self.model.body(name='table_2').pos.tolist()),
            "marker_2": str(self.model.body(name='table2_marker').pos.tolist()),
        }
        self.data_files['setup'].writerow(setup)

        target_1 = np.concatenate((self.model.body(name='target_1').pos.tolist(), self.model.body(name='target_1').quat.tolist()), axis=None)
        target_2 = np.concatenate((self.model.body(name='target_2').pos.tolist(), self.model.body(name='target_2').quat.tolist()), axis=None)

        step = {
            "timestamp" : timestamp,
            "theta": str(theta.tolist()),
            "thetadot": str(thetadot.tolist()),
            "target_1": str(target_1.tolist()),
            "target_2": str(target_2.tolist())
        }
        self.data_files['trajectory'].writerow(step)




def main(args=None):
    rclpy.init(args=args)
    visualizer = Visualizer()
    print("Initialized node.")

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("Node interrupted with Ctrl+C")
    finally:
        if PLAYBACK==False and USE_HARDWARE==True:
            visualizer.close_connection()
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()