import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

import os
import time
import json
import csv
import numpy as np


import mujoco
from mujoco import viewer

from sampling_based_planner.mpc_planner import run_cem_planner
from sampling_based_planner.quat_math import quaternion_distance, quaternion_multiply, rotation_quaternion, angle_between_lines_np, turn_quat


from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')

class Planner(Node):
    def __init__(self):
        super().__init__('planner')

        # Declare all parameters
        self.declare_parameter('use_hardware', False)
        self.declare_parameter('record_data', False)
        self.declare_parameter('idx', 0)
        self.declare_parameter('num_batch', 500)
        self.declare_parameter('num_steps', 15)
        self.declare_parameter('maxiter_cem', 1)
        self.declare_parameter('maxiter_projection', 5)
        self.declare_parameter('w_pos', 3.0)
        self.declare_parameter('w_rot', 0.5)
        self.declare_parameter('w_col', 500.0)
        self.declare_parameter('num_elite', 0.05)
        self.declare_parameter('timestep', 0.1)
        self.declare_parameter('position_threshold', 0.06)
        self.declare_parameter('rotation_threshold', 0.1)

        # Demo params
        self.use_hardware = self.get_parameter('use_hardware').get_parameter_value().bool_value
        self.record_data_ = self.get_parameter('record_data').get_parameter_value().bool_value
        self.idx = self.get_parameter('idx').get_parameter_value().integer_value
        self.idx = str(self.idx).zfill(3)

        # Planner params
        self.num_dof = 12
        self.init_joint_position = [1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0]
        # self.init_joint_position = np.array([ 1.45, -0.88,  1.75, -2.45, -1.6,  -0. ,  -1.88, -1.19,  1.75, -2.01, -1.6,   0.  ])
        num_batch = self.get_parameter('num_batch').get_parameter_value().integer_value
        num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        maxiter_cem = self.get_parameter('maxiter_cem').get_parameter_value().integer_value
        maxiter_projection = self.get_parameter('maxiter_projection').get_parameter_value().integer_value
        w_pos = self.get_parameter('w_pos').get_parameter_value().double_value
        w_rot = self.get_parameter('w_rot').get_parameter_value().double_value
        w_col = self.get_parameter('w_col').get_parameter_value().double_value
        num_elite = self.get_parameter('num_elite').get_parameter_value().double_value
        timestep = self.get_parameter('timestep').get_parameter_value().double_value
        position_threshold = self.get_parameter('position_threshold').get_parameter_value().double_value
        rotation_threshold = self.get_parameter('rotation_threshold').get_parameter_value().double_value


        self.task = 'pick'

        cost_weights = {
            'collision': w_col,
			'theta': 0.3,
			'z-axis': 5.0,
            'velocity': 0.1,

            'position': w_pos,
            'orientation': w_rot,

            'distance': 20.0,

            'pick': 0,
            'move': 0
        }

        # Open files to which the data will be saved
        if self.record_data_:
            self.pathes = {
                "setup": os.path.join(PACKAGE_DIR, 'data', 'planner', 'setup', f'setup_{self.idx}.csv'),
                "trajectory": os.path.join(PACKAGE_DIR, 'data', 'planner', 'trajectory', f'trajectory_{self.idx}.csv'),
            }
            self.data_files = dict()

            self.data_files['setup'] = csv.DictWriter(open(self.pathes['setup'], "w+"), fieldnames=['table_1', 'marker_1', 'table_2', 'marker_2'])
            self.data_files['trajectory'] = csv.DictWriter(open(self.pathes['trajectory'], "w+"), fieldnames=['timestamp', 'theta', 'thetadot', 'theta_horizon', 'thetadot_horizon', 'target_1', 'target_2'])

            self.data_files['setup'].writeheader()
            self.data_files['trajectory'].writeheader()

        self.grab_pos_thresh = 0.021
        self.grab_rot_thresh = 0.11

        # Initialize robot connection
        self.rtde_c_1 = None
        self.rtde_r_1 = None

        self.rtde_c_2 = None
        self.rtde_r_2 = None

        self.grippers = {
                            '1': {
                                'srv': None,
                                'state': 'open'
                            },
                            '2': {
                                'srv': None,
                                'state': 'open'
                            }
                        }


        if self.use_hardware:
            try:
                from gripper_srv.srv import GripperService

                self.rtde_c_1 = RTDEControl("192.168.0.120")
                self.rtde_r_1 = RTDEReceive("192.168.0.120")

                self.rtde_c_2 = RTDEControl("192.168.0.124")
                self.rtde_r_2 = RTDEReceive("192.168.0.124")

                self.grippers['1']['srv'] = self.create_client(GripperService, 'gripper_1/gripper_service')
                while not self.grippers['1']['srv'].wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('Gripper 1 service not available, waiting again...')

                self.grippers['2']['srv'] = self.create_client(GripperService, 'gripper_2/gripper_service')
                while not self.grippers['2']['srv'].wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('Gripper 2 service not available, waiting again...')

                self.req = GripperService.Request()
                print("Connection with UR5e established.", flush=True)
            except Exception as e:
                print(f"Could not connect to robot: {e}", flush=True)
                rclpy.shutdown()
                return

            # Move to initial position
            self.move_to_start()
        
        # Initialize MuJoCo model and data
        model_path = os.path.join(get_package_share_directory('real_demo'), 'ur5e_hande_mjx', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        # self.tray_idx = self.model.jnt_qposadr[self.model.body_jntadr[self.model.body(name="tray").id]]

        target_0_rot = quaternion_multiply(quaternion_multiply(self.model.body(name="target_0").quat, rotation_quaternion(-180, [0, 1, 0])), rotation_quaternion(-90, [0, 0, 1]))
        target_1_rot = quaternion_multiply(quaternion_multiply(self.model.body(name="target_1").quat, rotation_quaternion(180, [0, 1, 0])), rotation_quaternion(90, [0, 0, 1]))

        self.model.body(name='target_0').quat = target_0_rot
        self.model.body(name='target_1').quat = target_1_rot

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

        mujoco.mj_forward(self.model, self.data)

        # Set the table positions alligmed with the motion capture coordinate system
        if self.use_hardware:
            setup = list(csv.DictReader(open(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.csv'), "r")))[-1]
            table_1_pos = json.loads(setup['table_1'])
            table_2_pos = json.loads(setup['table_2'])
            self.model.body(name='table_1').pos = table_1_pos
            self.model.body(name='table1_marker').pos = json.loads(setup['marker_1'])
            self.model.body(name='table_2').pos = table_2_pos
            self.model.body(name='table2_marker').pos = json.loads(setup['marker_2'])
        else:
            table_1_pos = self.model.body(name='table_1').pos
            table_2_pos = self.model.body(name='table_2').pos
        
        # Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self.model,
            data=self.data,
            num_dof=self.num_dof,
            num_batch=num_batch,
            num_steps=num_steps,
            maxiter_cem=maxiter_cem,
            maxiter_projection=maxiter_projection,
            num_elite=num_elite,
            timestep=timestep,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            table_1_pos=table_1_pos,
            table_2_pos=table_2_pos,
            cost_weights=cost_weights
        )

        self.eef_pos_1_init = self.data.site_xpos[self.planner.tcp_id_1].copy()
        self.eef_pos_2_init = self.data.site_xpos[self.planner.tcp_id_2].copy()
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        if self.use_hardware:
            self.viewer.cam.lookat[:] = [-3.5, 0.0, 0.8]     
        else:
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        # Setup subscribers
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.subscription_object1 = self.create_subscription(PoseStamped, '/vrpn_mocap/object1/pose', self.object1_callback, qos_profile)
        self.subscription_object1 = self.create_subscription(PoseStamped, '/vrpn_mocap/object2/pose', self.object2_callback, qos_profile)
        self.subscription_obstacle1 = self.create_subscription(PoseStamped, '/vrpn_mocap/obstacle1/pose', self.obstacle1_callback, qos_profile)
        
        # Setup control timer
        self.timer = self.create_timer(timestep, self.control_loop)

    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()

        if self.task == 'move':
            eef_pos_1 = self.data.site_xpos[self.planner.tcp_id_1]
            eef_pos_2 = self.data.site_xpos[self.planner.tcp_id_2]

            tray_pos = (eef_pos_1+eef_pos_2)/2 - np.array([0, 0, 0.1])
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos

            tray_rot_init = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
            tray_site_1 = self.data.site_xpos[self.model.site(name="tray_site_1").id]
            tray_site_2 = self.data.site_xpos[self.model.site(name="tray_site_2").id]
            tray_rot = turn_quat(tray_site_1, tray_site_2, eef_pos_1, eef_pos_2, tray_rot_init)
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_rot
            # print(tray_pos, tray_rot_init, tray_rot, flush=True)

            
        # self.eef_pos_1_init = self.data.site_xpos[self.planner.tcp_id_1].copy()
        # self.eef_pos_2_init = self.data.site_xpos[self.planner.tcp_id_2].copy()
        if self.use_hardware:
            # Get current state
            current_pos_1 = np.array(self.rtde_r_1.getActualQ())
            current_vel_1 = np.array(self.rtde_r_1.getActualQd())

            current_pos_2 = np.array(self.rtde_r_2.getActualQ())
            current_vel_2 = np.array(self.rtde_r_2.getActualQd())

            current_pos = np.concatenate((current_pos_1, current_pos_2), axis=None)
            current_vel = np.concatenate((current_vel_1, current_vel_2), axis=None)
        else:
            current_pos = self.data.qpos[self.joint_mask_pos]
            current_vel = self.data.qvel[self.joint_mask_vel]
        
        
        # Compute control
        thetadot, cost, cost_list, thetadot_horizon, theta_horizon = self.planner.compute_control(current_pos, current_vel, self.task)
        
        if self.use_hardware:
            # Send velocity command
            self.rtde_c_1.speedJ(thetadot[:self.planner.num_dof//2], acceleration=1.4, time=0.1)
            self.rtde_c_2.speedJ(thetadot[self.planner.num_dof//2:], acceleration=1.4, time=0.1)

            # Update MuJoCo state
            current_pos = np.concatenate((np.array(self.rtde_r_1.getActualQ()), np.array(self.rtde_r_2.getActualQ())), axis=None)
            self.data.qpos[self.joint_mask_pos] = current_pos
            mujoco.mj_forward(self.model, self.data)
        else:
            self.data.qvel[:] = np.zeros(len(self.joint_mask_vel))
            self.data.qvel[self.joint_mask_vel] = thetadot
            mujoco.mj_step(self.model, self.data)

        current_cost_g_1 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_1] - self.planner.target_1[:3])
        current_cost_r_1 = quaternion_distance(self.data.xquat[self.planner.hande_id_1], self.planner.target_1[3:])
            
        current_cost_g_2 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_2] - self.planner.target_2[:3])
        current_cost_r_2 = quaternion_distance(self.data.xquat[self.planner.hande_id_2], self.planner.target_2[3:])

        target_reached = (
                current_cost_g_1 < self.grab_pos_thresh \
                and current_cost_r_1 < self.grab_rot_thresh \
                and current_cost_g_2 < self.grab_pos_thresh \
                and current_cost_r_2 < self.grab_rot_thresh
        )
        if target_reached and self.task=='pick':
            self.task = 'move'


        # if self.task == 'pick':
        #     if current_cost_g_1 < self.grab_pos_thresh and current_cost_r_1 < self.grab_rot_thresh:
        #         if self.task == 'pick' and self.grippers['1']['state']=='open':
        #             self.gripper_control(gripper_idx=1, action='close')
        #         # elif self.task == 'move' and self.grippers['1']['state']=='close':
        #         #     self.gripper_control(gripper_idx=1, action='open') 
            
        #     if current_cost_g_2 < self.grab_pos_thresh and current_cost_r_2 < self.grab_rot_thresh:
        #         if self.task == 'pick' and self.grippers['2']['state']=='open':
        #             self.gripper_control(gripper_idx=2, action='close') 
        #         # elif self.task == 'move' and self.grippers['2']['state']=='close':
        #         #     self.gripper_control(gripper_idx=2, action='open') 

        if self.record_data_:
            theta = self.data.qpos[self.joint_mask_pos]
            self.record_data(theta=theta, thetadot=thetadot, theta_horizon=theta_horizon, thetadot_horizon=thetadot_horizon)
        
        # Update viewer
        self.viewer.sync()

        cost_c, cost_dist, cost_g, cost_r = cost_list
        
        # Print debug info
        print(f'Task: {self.task} | '
              f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | '
              f'Cost dist: {"%.2f"%(float(cost_dist))} | '
              f'Cost g: {"%.2f"%(float(cost_g))} | '
              f'Cost r: {"%.2f"%(float(cost_r))} | '
              f'Cost c: {"%.2f"%(float(cost_c))} | '
              f'Cost gr1: {"%.2f, %.2f"%(float(current_cost_g_1), float(current_cost_r_1))} | '
              f'Cost gr2: {"%.2f, %.2f"%(float(current_cost_g_2), float(current_cost_r_2))} | '
              f'Cost: {np.round(cost, 2)}', flush=True)
        
        # print(self.data.qpos[self.tray_idx+3:self.tray_idx+7], self.model.body(name="target_2").quat)
        
    def gripper_control(self, gripper_idx=1, action='open'):

        self.data.ctrl[gripper_idx-1] = 255 if action == 'close' else 0

        if self.use_hardware:
            self.req.position = 100 if action == 'close' else 0
            self.req.speed = 255
            self.req.force = 255
            resp = self.grippers[str(gripper_idx)]['srv'].call_async(self.req)

        self.grippers[str(gripper_idx)]['state'] = action

        if self.grippers['1']['state'] == 'close' and self.grippers['2']['state'] == 'close' and self.task == 'pick':
            self.task = 'move'
        
        print(f"Gripper {gripper_idx} has complited {action} action.")

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.rtde_c_1.moveJ(self.init_joint_position[:self.num_dof//2], asynchronous=False)
        self.rtde_c_2.moveJ(self.init_joint_position[self.num_dof//2:], asynchronous=False)
        self.gripper_control(gripper_idx=1, action='open') 
        self.gripper_control(gripper_idx=2, action='open') 
        print("Moved to initial pose.", flush=True)

    def close_connection(self):
        if self.use_hardware:
            """Cleanup robot connection"""
            if self.rtde_c_1:
                self.rtde_c_1.speedStop()
                self.rtde_c_1.disconnect()

            if self.rtde_c_2:
                self.rtde_c_2.speedStop()
                self.rtde_c_2.disconnect()
            print("Disconnected from UR5e Robot", flush=True)

    def object1_callback(self, msg):
        """Callback for target object pose updates"""
        pose = msg.pose
        target_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z+0.1])
        target_rot = np.array([0.0, 1.0, 0, 0])
        
        self.model.body(name='target_0').pos = target_pos
        self.model.body(name='target_0').quat = target_rot
        self.planner.update_targets(target_idx=1, target_pos=target_pos, target_rot = target_rot)

    def object2_callback(self, msg):
        """Callback for target object pose updates"""
        pose = msg.pose
        target_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z+0.1])
        target_rot = np.array([0.0, 1.0, 0, 0])
        
        self.model.body(name='target_1').pos = target_pos
        self.model.body(name='target_1').quat = target_rot
        self.planner.update_targets(target_idx=2, target_pos=target_pos, target_rot = target_rot)

    def obstacle1_callback(self, msg):
        """Callback for obstacle pose updates"""
        pose = msg.pose
        obstacle_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z])
        obstacle_rot = np.array([0.0, 1.0, 0, 0])
        self.planner.update_obstacle(obstacle_pos, obstacle_rot)
    
        
    def record_data(self, theta, thetadot, theta_horizon, thetadot_horizon):
        """Save data to csv file"""
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
            "theta_horizon": str(theta_horizon.tolist()),
            "thetadot_horizon": str(thetadot_horizon.tolist()),
            "target_1": str(target_1.tolist()),
            "target_2": str(target_2.tolist())
        }
        self.data_files['trajectory'].writerow(step)

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    print("Initialized node.", flush=True)
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        if rclpy.ok():
            planner.close_connection()
            planner.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()