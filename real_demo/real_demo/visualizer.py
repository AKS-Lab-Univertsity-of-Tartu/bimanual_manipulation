import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
import json

import time

import mujoco
from mujoco import viewer
import numpy as np

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')

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

        self.pathes = {
            "setup": os.path.join(PACKAGE_DIR, 'json', 'setup', f'setup_{idx}.json'),
            "trajectory": os.path.join(PACKAGE_DIR, 'json', 'trajectory', f'trajectory_{idx}.json'),
        }
        if RECORD_DATA:
            self.json_files = dict()
            for key, value in self.pathes.items():
                self.json_files[key] = open(value, "w+")
        elif PLAYBACK:
            self.json_files = dict()
            for key, value in self.pathes.items():
                self.json_files[key] = open(value, "r")


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
            self.setup = json.load(self.json_files['setup'])

            self.model.body(name='table_1').pos = self.setup['table1']
            self.model.body(name='table1_marker').pos = self.setup['marker1']
            self.model.body(name='table_2').pos = self.setup['table2']
            self.model.body(name='table2_marker').pos = self.setup['marker2']

            self.trajectory = json.load(self.json_files['trajectory'])
            self.step_idx = 0
            self.timer = self.create_timer(0.05, self.view_playback)

    def view_model(self):
        step_start = time.time()
        theta_1 = self.rtde_r_1.getActualQ()
        theta_2 = self.rtde_r_2.getActualQ()

        theta = np.concatenate((theta_1, theta_2), axis=None)

        self.data.qpos[:12] = theta

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        if RECORD_DATA:
            self.record_data(theta=theta)

        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)   

    def view_playback(self):
        step_start = time.time()

        theta = self.trajectory[self.step_idx]['theta']

        self.model.body(name='target_1').pos = self.trajectory[self.step_idx]['target_1']['position']
        self.model.body(name='target_1').quat = self.trajectory[self.step_idx]['target_1']['orientation']

        self.model.body(name='target_2').pos = self.trajectory[self.step_idx]['target_2']['position']
        self.model.body(name='target_2').quat = self.trajectory[self.step_idx]['target_2']['orientation']

        self.data.qpos[:12] = theta

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        self.step_idx+=1

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

    def record_data(self, theta):
        timestep = time.time()

        setup = {
            "table1": self.model.body(name='table_1').pos.tolist(),
            "table2": self.model.body(name='table_2').pos.tolist(),
            "marker1": self.model.body(name='table1_marker').pos.tolist(),
            "marker2": self.model.body(name='table2_marker').pos.tolist(),
        }

        self.json_files['setup'].seek(0)        # Go to start of file
        self.json_files['setup'].truncate()     # Clear existing content
        json.dump(setup, self.json_files['setup'], indent=2)
        self.json_files['setup'].flush()  

        step = {
            "timestep" : timestep,
            "theta": theta.tolist(),
            "target_1": {
                "position": self.model.body(name='target_1').pos.tolist(),
                "orientation": self.model.body(name='target_1').quat.tolist(),
            },
            "target_2": {
                "position": self.model.body(name='target_2').pos.tolist(),
                "orientation": self.model.body(name='target_2').quat.tolist(),
            }
        }
        self.trajectory.append(step)

        self.json_files['trajectory'].seek(0)        # Go to start of file
        self.json_files['trajectory'].truncate()     # Clear existing content
        json.dump(self.trajectory, self.json_files['trajectory'], indent=2)
        self.json_files['trajectory'].flush()  




def main(args=None):
    rclpy.init(args=args)
    visualizer = Visualizer()
    print("Initialized node.")

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("Node interrupted with Ctrl+C")
    finally:
        if not PLAYBACK:
            visualizer.close_connection()
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()