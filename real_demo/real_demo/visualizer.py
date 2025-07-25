import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

import os
import json
import csv
import time

import mujoco
from mujoco import viewer
import numpy as np

from sampling_based_planner.quat_math import quaternion_distance, quaternion_multiply, rotation_quaternion, angle_between_lines_np, turn_quat

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')

        self.declare_parameter('use_hardware', False)
        self.declare_parameter('record_data', False)
        self.declare_parameter('playback', True)
        self.declare_parameter('folder', "manual") # manual or planner
        self.declare_parameter('idx', 0)

        self.use_hardware = self.get_parameter('use_hardware').get_parameter_value().bool_value
        self.record_data_ = self.get_parameter('record_data').get_parameter_value().bool_value
        self.playback = self.get_parameter('playback').get_parameter_value().bool_value
        self.folder = self.get_parameter('folder').get_parameter_value().string_value
        self.idx = self.get_parameter('idx').get_parameter_value().integer_value
        self.idx = str(self.idx).zfill(3)

        self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
        self.trajectory = list()
        self.num_dof = 12
        self.num_steps = 15

        model_path = os.path.join(PACKAGE_DIR, 'ur5e_hande_mjx', 'scene.xml')

        self.pathes = {
            "setup": os.path.join(PACKAGE_DIR, 'data', self.folder, 'setup', f'setup_{self.idx}.csv'),
            "trajectory": os.path.join(PACKAGE_DIR, 'data', self.folder, 'trajectory', f'trajectory_{self.idx}.csv'),
        }
        if self.record_data_:
            self.data_files = dict()
            self.data_files['setup'] = csv.DictWriter(open(self.pathes['setup'], "w+"), fieldnames=['table_1', 'marker_1', 'table_2', 'marker_2'])
            self.data_files['trajectory'] = csv.DictWriter(open(self.pathes['trajectory'], "w+"), fieldnames=['timestamp', 'theta', 'thetadot', 'target_1', 'target_2'])
            self.data_files['setup'].writeheader()
            self.data_files['trajectory'].writeheader()

        elif self.playback:
            self.data_files = dict()
            for key, value in self.pathes.items():
                self.data_files[key] = list(csv.DictReader(open(value, "r")))


        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.1

        joint_names_pos = list()
        joint_names_vel = list()
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            
            for _ in range(n_pos):
                joint_names_pos.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        
        
        robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
                        'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        
        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)


        self.data = mujoco.MjData(self.model)

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position
        self.init_tray_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]].copy()
        self.init_tray_quat = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]].copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        if self.use_hardware:
            self.viewer.cam.lookat[:] = [-3.0, 0.0, 0.8]     
        else:
            self.viewer.cam.lookat[:] = [-0.0, 0.0, 0.8] #[0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        if not self.playback:

            if self.use_hardware:
                self.rtde_c_1 = RTDEControl("192.168.0.120")
                self.rtde_r_1 = RTDEReceive("192.168.0.120")

                self.rtde_c_2 = RTDEControl("192.168.0.124")
                self.rtde_r_2 = RTDEReceive("192.168.0.124")
                self.move_to_start()

            qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
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

            self.timer = self.create_timer(self.model.opt.timestep, self.view_model)
        else:
            self.setup = self.data_files['setup'][-1]

            self.model.body(name='table_1').pos = json.loads(self.setup['table_1'])
            self.model.body(name='table1_marker').pos = json.loads(self.setup['marker_1'])
            self.model.body(name='table_2').pos = json.loads(self.setup['table_2'])
            self.model.body(name='table2_marker').pos = json.loads(self.setup['marker_2'])

            self.tcp_id_1 = self.model.site(name="tcp").id
            self.hande_id_1 = self.model.body(name="hande").id
            self.tcp_id_2 = self.model.site(name="tcp_2").id
            self.hande_id_2 = self.model.body(name="hande_2").id

            self.eef_pos_1_init = self.data.site_xpos[self.tcp_id_1].copy()
            self.eef_pos_2_init = self.data.site_xpos[self.tcp_id_2].copy()
            self.tray_site_1_init = self.data.site_xpos[self.model.site(name="tray_site_1").id].copy()
            self.tray_site_2_init = self.data.site_xpos[self.model.site(name="tray_site_2").id].copy()

            self.step_idx = 0
            self.timer = self.create_timer(self.model.opt.timestep, self.view_playback)

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.rtde_c_1.moveJ(self.init_joint_position[:self.num_dof//2], asynchronous=False)
        self.rtde_c_2.moveJ(self.init_joint_position[self.num_dof//2:], asynchronous=False)
        print("Moved to initial pose.")

    def view_model(self):
        step_start = time.time()

        if self.use_hardware:
            theta_1 = self.rtde_r_1.getActualQ()
            theta_2 = self.rtde_r_2.getActualQ()
            theta = np.concatenate((theta_1, theta_2), axis=None)

            thetadot_1 = self.rtde_r_1.getActualQd()
            thetadot_2 = self.rtde_r_2.getActualQd()
            thetadot = np.concatenate((thetadot_1, thetadot_2), axis=None)
        else:
            theta = self.data.qpos[self.joint_mask_pos]
            thetadot = self.data.qvel[self.joint_mask_vel]

        self.data.qpos[self.joint_mask_pos] = theta

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        if self.record_data_:
            self.record_data(theta=theta, thetadot=thetadot)

        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)   

    def view_playback(self):
        step_start = time.time()

        if self.folder=="manual":
            theta = json.loads(self.data_files['trajectory'][self.step_idx]['theta'])
            self.data.qpos[self.joint_mask_pos] = theta
        elif self.folder=="planner":
            theta = json.loads(self.data_files['trajectory'][self.step_idx]['theta'])
            # self.data.qvel[:self.num_dof] = np.mean(thetadot_horizon[1:int(self.num_steps*0.9)], axis=0)
            # print(thetadot, flush=True)
            self.data.qpos[self.joint_mask_pos] = theta

        if self.step_idx >= 50:
            eef_pos_1 = self.data.site_xpos[self.tcp_id_1]
            eef_pos_2 = self.data.site_xpos[self.tcp_id_2]

            tray_pos = (eef_pos_1+eef_pos_2)/2 - np.array([0, 0, 0.1])
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos

            tray_rot_init = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]

            tray_site_1 = self.data.site_xpos[self.model.site(name="tray_site_1").id].copy()
            tray_site_2 = self.data.site_xpos[self.model.site(name="tray_site_2").id].copy()

            print(tray_site_1, eef_pos_1, flush=True)

            tray_rot = turn_quat(tray_site_2, tray_site_1, eef_pos_1, eef_pos_2, tray_rot_init)
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_rot

            print(tray_pos, tray_rot_init, tray_rot, flush=True)

        self.eef_pos_1_init = self.data.site_xpos[self.tcp_id_1].copy()
        self.eef_pos_2_init = self.data.site_xpos[self.tcp_id_2].copy()

        print(self.step_idx, flush=True)

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        if self.step_idx < len(self.data_files['trajectory'])-1:
            self.step_idx += 1
        else:
            # print("restart", flush=True)
            self.step_idx = 0
            self.data.qpos[self.joint_mask_pos] = self.init_joint_position
            self.data.qvel[self.joint_mask_vel] = np.zeros(self.init_joint_position.shape)
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = self.init_tray_pos
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = self.init_tray_quat
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            

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
        self.model.body(name='target_0').pos = marker_pose

    def object2_callback(self, msg):
        marker_pose =  [-msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z]
        self.model.body(name='target_1').pos = marker_pose

    def close_connection(self):
        if self.playback==False and self.use_hardware==True:
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

        target_1 = np.concatenate((self.model.body(name='target_0').pos.tolist(), self.model.body(name='target_0').quat.tolist()), axis=None)
        target_2 = np.concatenate((self.model.body(name='target_1').pos.tolist(), self.model.body(name='target_1').quat.tolist()), axis=None)

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
        visualizer.close_connection()
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()