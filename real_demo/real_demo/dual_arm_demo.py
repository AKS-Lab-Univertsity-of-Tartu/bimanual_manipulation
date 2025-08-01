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
from sampling_based_planner.quat_math import *

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')
np.set_printoptions(precision=4, suppress=True)

target_positions = np.array([
    [-0.25, -0.2, 0.3],
    [-0.25, 0.0, 0.3],
    [-0.25, -0.1, 0.3],
])

target_rotations = np.array([
    quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-90, [0, 0, 1])),
    quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-135, [0, 0, 1])),
    quaternion_multiply(np.array([1, 0, 0, 0]), rotation_quaternion(-45, [0, 0, 1])),
])

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
        num_batch = self.get_parameter('num_batch').get_parameter_value().integer_value
        num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        maxiter_cem = self.get_parameter('maxiter_cem').get_parameter_value().integer_value
        maxiter_projection = self.get_parameter('maxiter_projection').get_parameter_value().integer_value
        w_pos = self.get_parameter('w_pos').get_parameter_value().double_value
        w_rot = self.get_parameter('w_rot').get_parameter_value().double_value
        w_col = self.get_parameter('w_col').get_parameter_value().double_value
        num_elite = self.get_parameter('num_elite').get_parameter_value().double_value
        self.timestep = self.get_parameter('timestep').get_parameter_value().double_value
        position_threshold = self.get_parameter('position_threshold').get_parameter_value().double_value
        rotation_threshold = self.get_parameter('rotation_threshold').get_parameter_value().double_value

        if self.record_data_:
            self.pathes = {
                "setup": os.path.join(PACKAGE_DIR, 'data', 'planner', 'setup', f'setup_{self.idx}.npz'),
                "trajectory": os.path.join(PACKAGE_DIR, 'data', 'planner', 'trajectory', f'traj_{self.idx}.npz'),
            }

            # Store data in lists during runtime
            self.data_buffers = {
                'setup': [],
                'theta': [],
                'thetadot': [],
                'theta_planned': [],
                'thetadot_planned': [],
                'target_0': [],
                'target_1': [],
                'theta_planned_batched': [],
                'thetadot_planned_batched': [],
                'cost_cgr_batched': [],
                'timestamp': [],
            }

        self.task = 'pick'
        self.target_idx = 0

        cost_weights = {
            'collision': 500,
			'theta': 0.3,
			'z-axis': 5.0,
            'velocity': 0.1,

            'position': 3.0,
            'orientation_pick': 0.5,

            'distance': 20.0,
            'position_tray': 3.0,
            'orientation_tray': 2,
            'orientation_move': 10,

            'pick': 0,
            'move': 0
        }

        self.grab_pos_thresh = 0.02
        self.grab_rot_thresh = 0.05
        self.thetadot = np.zeros(self.num_dof)

        # Initialize robot connection
        self.rtde_c_0 = None
        self.rtde_r_0 = None

        self.rtde_c_1 = None
        self.rtde_r_1 = None

        self.grippers = {
            '0': {
                'srv': None,
                'state': 'open'
            },
            '1': {
                'srv': None,
                'state': 'open'
            }
        }

        if self.use_hardware:
            self.initialize_robot_connection()
        
        # Initialize MuJoCo model and data
        model_path = os.path.join(get_package_share_directory('real_demo'), 'ur5e_hande_mjx', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.timestep

        self.data = mujoco.MjData(self.model)
  
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

        self.data.qpos[self.joint_mask_pos] = self.init_joint_position

        target_0_rot = quaternion_multiply(quaternion_multiply(self.model.body(name="target_0").quat, rotation_quaternion(-180, [0, 1, 0])), rotation_quaternion(-90, [0, 0, 1]))
        target_1_rot = quaternion_multiply(quaternion_multiply(self.model.body(name="target_1").quat, rotation_quaternion(180, [0, 1, 0])), rotation_quaternion(90, [0, 0, 1]))
        target_2_rot = quaternion_multiply(self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]], rotation_quaternion(-45, [0, 0, 1]))

        self.model.body(name='target_0').quat = target_0_rot
        self.model.body(name='target_1').quat = target_1_rot
        self.model.body(name='target_00').quat = target_0_rot
        self.model.body(name='target_11').quat = target_1_rot
        self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] = target_2_rot

        # Set the table positions alligmed with the motion capture coordinate system
        if self.use_hardware:
            setup = np.load(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.npz'), allow_pickle=True)

            marker_pos = setup['setup'][0][1]
            marker_diff = marker_pos-self.model.body(name='table0_marker').pos

            self.model.body(name='table_0').pos = setup['setup'][0][0]
            self.model.body(name='table0_marker').pos = setup['setup'][0][1]
            self.model.body(name='table_1').pos = setup['setup'][0][2]
            self.model.body(name='table1_marker').pos = setup['setup'][0][3]

            self.model.body(name='tray').pos += marker_diff
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] += marker_diff
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] += marker_diff
        else:
            table_0_pos = self.model.body(name='table_0').pos
            table_1_pos = self.model.body(name='table_1').pos

        mujoco.mj_forward(self.model, self.data)

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
            timestep=self.timestep,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            table_0_pos=table_0_pos,
            table_1_pos=table_1_pos,
            cost_weights=cost_weights
        )
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = self.model.body(name='table_0').pos
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        # Setup subscribers
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.subscription_object0 = self.create_subscription(PoseStamped, '/vrpn_mocap/object1/pose', self.object0_callback, qos_profile)
        self.subscription_obstacle0 = self.create_subscription(PoseStamped, '/vrpn_mocap/obstacle1/pose', self.obstacle0_callback, qos_profile)
        
        # Start control timer
        self.timer = self.create_timer(self.timestep, self.control_loop)

    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()

        if self.task == 'move':
            eef_pos_0 = self.data.site_xpos[self.planner.tcp_id_0]
            eef_pos_1 = self.data.site_xpos[self.planner.tcp_id_1]

            tray_pos = (eef_pos_0+eef_pos_1)/2 - np.array([0, 0, 0.1])
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos

            tray_rot_init = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
            tray_0_pos = self.data.xpos[self.model.body(name='target_0').id]
            tray_1_pos = self.data.xpos[self.model.body(name='target_1').id]
            tray_rot = turn_quat(tray_0_pos, tray_1_pos, eef_pos_0, eef_pos_1, tray_rot_init)
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_rot

            self.planner.update_targets(target_idx=0, target_pos=self.data.xpos[self.model.body(name="target_00").id], target_rot=self.data.xquat[self.model.body(name="target_00").id])
            self.planner.update_targets(target_idx=1, target_pos=self.data.xpos[self.model.body(name="target_11").id], target_rot=self.data.xquat[self.model.body(name="target_11").id])

        # Get current state
        if self.use_hardware:
            current_pos_0 = np.array(self.rtde_r_0.getActualQ())
            current_pos_1 = np.array(self.rtde_r_1.getActualQ())

            current_pos = np.concatenate((current_pos_0, current_pos_1), axis=None)
            current_vel = self.thetadot
        else:
            current_pos = self.data.qpos[self.joint_mask_pos]
            current_vel = self.thetadot
        
        
        # Compute control
        self.thetadot, cost, cost_list, thetadot_horizon, theta_horizon = self.planner.compute_control(current_pos, current_vel, self.task)
        
        if self.use_hardware:
            # Send velocity command
            self.rtde_c_0.speedJ(self.thetadot[:self.planner.num_dof//2], acceleration=1, time=0.1)
            self.rtde_c_1.speedJ(self.thetadot[self.planner.num_dof//2:], acceleration=1, time=0.1)

            # Update MuJoCo state
            current_pos = np.concatenate((np.array(self.rtde_r_0.getActualQ()), np.array(self.rtde_r_1.getActualQ())), axis=None)
            self.data.qpos[self.joint_mask_pos] = current_pos
            mujoco.mj_forward(self.model, self.data)
        else:
            self.data.qvel[:] = np.zeros(len(self.joint_mask_vel))
            self.data.qvel[self.joint_mask_vel] = self.thetadot
            mujoco.mj_step(self.model, self.data)

        current_cost_g_0 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_0] - self.planner.target_0[:3])
        current_cost_r_0 = quaternion_distance(self.data.xquat[self.planner.hande_id_0], self.planner.target_0[3:])
            
        current_cost_g_1 = np.linalg.norm(self.data.site_xpos[self.planner.tcp_id_1] - self.planner.target_1[:3])
        current_cost_r_1 = quaternion_distance(self.data.xquat[self.planner.hande_id_1], self.planner.target_1[3:])

        tray_pos = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
        tray_rot = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap').id]]
        current_cost_g_tray = np.linalg.norm(tray_pos - self.planner.target_2[:3])
        current_cost_r_tray = quaternion_distance(tray_rot, self.planner.target_2[3:])

        if self.task=='pick':
            target_reached = (
                    current_cost_g_0 < self.grab_pos_thresh \
                    and current_cost_r_0 < self.grab_rot_thresh \
                    and current_cost_g_1 < self.grab_pos_thresh \
                    and current_cost_r_1 < self.grab_rot_thresh
            )
        elif self.task=='move':
            target_reached = (
                    current_cost_g_tray < self.grab_pos_thresh \
                    and current_cost_r_tray < self.grab_rot_thresh \
            )

        if target_reached and self.task=='pick':
            self.task = 'move'
            self.gripper_control(gripper_idx=0, action='close')
            self.gripper_control(gripper_idx=1, action='close')
        elif target_reached and self.task=='move':
            print("================== TARGRT REACHED UPDATING TARGET ==================")
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] = target_positions[self.target_idx]
            self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] = target_rotations[self.target_idx]
            self.planner.target_2[:3] = target_positions[self.target_idx]
            self.planner.target_2[3:] = target_rotations[self.target_idx]
            if self.target_idx < len(target_positions)-1:
                self.target_idx+=1

        if self.record_data_:    
            theta = self.data.qpos[self.joint_mask_pos]
            self.data_buffers['theta'].append(theta.copy())
            self.data_buffers['thetadot'].append(self.thetadot.copy())
            self.data_buffers['theta_planned'].append(theta_horizon.copy())
            self.data_buffers['thetadot_planned'].append(thetadot_horizon.copy())
            self.data_buffers['target_0'].append(self.planner.target_0.copy())
            self.data_buffers['target_1'].append(self.planner.target_1.copy())
            # self.data_buffers['theta_planned_batched'].append(th_batch[0].reshape((self.num_batch, self.num_dof, self.num_steps)).copy())
            # self.data_buffers['thetadot_planned_batched'].append(thd_batch[0].reshape((self.num_batch, self.num_dof, self.num_steps)).copy())
            # self.data_buffers['cost_cgr_batched'].append(cost_list_batch[0].copy())
            self.data_buffers['timestamp'].append(time.time())
        
        # Update viewer
        self.viewer.sync()

        cost_c, cost_dist, cost_g, cost_r = cost_list
        
        # Print debug info
        print(f'\n| Task: {self.task} '
              f'\n| Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms '
              f'\n| Cost eq: {"%.2f"%(float(cost_dist))} '
              f'\n| Cost g: {"%.2f"%(float(cost_g))} '
              f'\n| Cost r: {"%.2f"%(float(cost_r))} '
              f'\n| Cost c: {"%.2f"%(float(cost_c))} '
              f'\n| Cost gr0: {"%.2f, %.2f"%(float(current_cost_g_0), float(current_cost_r_0))} '
              f'\n| Cost gr1: {"%.2f, %.2f"%(float(current_cost_g_1), float(current_cost_r_1))} '
              f'\n| Cost tr: {"%.2f, %.2f"%(float(current_cost_g_tray), float(current_cost_r_tray))} '
              f'\n| Cost: {np.round(cost, 2)} ', flush=True)
        
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step) 
                
    def gripper_control(self, gripper_idx=0, action='open'):
        if self.use_hardware:
            self.req.position = 250 if action == 'close' else 0
            self.req.speed = 255
            self.req.force = 255
            resp = self.grippers[str(gripper_idx)]['srv'].call_async(self.req)

        self.grippers[str(gripper_idx)]['state'] = action
        
        print(f"Gripper {gripper_idx} has complited {action} action.")

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.rtde_c_1.moveJ(self.init_joint_position[self.num_dof//2:], asynchronous=False)
        self.rtde_c_0.moveJ(self.init_joint_position[:self.num_dof//2], asynchronous=False)
        self.gripper_control(gripper_idx=0, action='open') 
        self.gripper_control(gripper_idx=1, action='open') 
        print("Moved to initial pose.", flush=True)

    def initialize_robot_connection(self):
        try:
            from gripper_srv.srv import GripperService

            self.rtde_c_0 = RTDEControl("192.168.0.120")
            self.rtde_r_0 = RTDEReceive("192.168.0.120")

            self.rtde_c_1 = RTDEControl("192.168.0.124")
            self.rtde_r_1 = RTDEReceive("192.168.0.124")

            self.grippers['0']['srv'] = self.create_client(GripperService, 'gripper_0/gripper_service')
            while not self.grippers['0']['srv'].wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Gripper 0 service not available, waiting again...')

            self.grippers['1']['srv'] = self.create_client(GripperService, 'gripper_1/gripper_service')
            while not self.grippers['1']['srv'].wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Gripper 1 service not available, waiting again...')

            self.req = GripperService.Request()
            print("Connection with UR5e established.", flush=True)
        except Exception as e:
            print(f"Could not connect to robot: {e}", flush=True)
            rclpy.shutdown()
            return

        # Move to initial position
        self.move_to_start()

    def close_connection(self):
        if self.use_hardware:
            """Cleanup robot connection"""
            if self.rtde_c_0:
                self.rtde_c_0.speedStop()
                self.rtde_c_0.disconnect()

            if self.rtde_c_1:
                self.rtde_c_1.speedStop()
                self.rtde_c_1.disconnect()
            print("Disconnected from UR5e Robot", flush=True)

    def object0_callback(self, msg):
        """Callback for target object pose updates"""

        if self.task == 'pick':
            pose = msg.pose
            tray_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z-0.08])
            self.model.body(name='tray').pos = tray_pos
            self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap').id]] = tray_pos
            mujoco.mj_forward(self.model, self.data)
            self.planner.update_targets(target_idx=0, target_pos=self.data.xpos[self.model.body(name="target_0").id], target_rot = self.model.body(name='target_0').quat)
            self.planner.update_targets(target_idx=1, target_pos=self.data.xpos[self.model.body(name="target_1").id], target_rot = self.model.body(name='target_1').quat)

    def obstacle0_callback(self, msg):
        """Callback for obstacle pose updates"""
        pose = msg.pose
        obstacle_pos = np.array([-pose.position.x, -pose.position.y, pose.position.z])
        obstacle_rot = np.array([0.0, 1.0, 0, 0])
        self.planner.update_obstacle(obstacle_pos, obstacle_rot)

    def record_data(self):
        """Save data to npy file"""
        self.data_buffers['setup'].append([self.model.body(name='table_0').pos, self.model.body(name='table0_marker').pos, 
                                           self.model.body(name='table_1').pos, self.model.body(name='table1_marker').pos])
        np.savez(
            self.pathes['setup'],
            setup=self.data_buffers['setup'],
        )
        np.savez(
            self.pathes['trajectory'],
            theta=np.array(self.data_buffers['theta']),
            thetadot=np.array(self.data_buffers['thetadot']),
            theta_planned=np.array(self.data_buffers['theta_planned']),
            thetadot_planned=np.array(self.data_buffers['thetadot_planned']),
            target_0=np.array(self.data_buffers['target_0']),
            target_1=np.array(self.data_buffers['target_1']),
            theta_planned_batched=np.array(self.data_buffers['theta_planned_batched']),
            thetadot_planned_batched=np.array(self.data_buffers['thetadot_planned_batched']),
            cost_cgr_batched=np.array(self.data_buffers['cost_cgr_batched']),
            timestamp=np.array(self.data_buffers['timestamp']),
        )
        self.data_saved = True
        print("Saving data...")

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    print("Initialized node.", flush=True)
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        # if rclpy.ok():
        if planner.record_data_:
            planner.record_data()
        planner.close_connection()
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()