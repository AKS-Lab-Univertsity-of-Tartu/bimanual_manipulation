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

from sampling_based_planner.mpc_planner_2 import run_cem_planner
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

PACKAGE_DIR = get_package_share_directory('real_demo')

idx = '2'.zfill(3)


class MocapListener(Node):
    def __init__(self):
        super().__init__('mocap_listener')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.use_hardware = True
        self.record_data_ = True

        if self.record_data_:
            self.pathes = {
                "setup": os.path.join(PACKAGE_DIR, 'data', 'planner', 'setup', f'setup_{idx}.csv'),
                "trajectory": os.path.join(PACKAGE_DIR, 'data', 'planner', 'trajectory', f'trajectory_{idx}.csv'),
            }
            self.data_files = dict()
            self.data_files['setup'] = csv.DictWriter(open(self.pathes['setup'], "w+"), fieldnames=['table_1', 'marker_1', 'table_2', 'marker_2'])
            self.data_files['trajectory'] = csv.DictWriter(open(self.pathes['trajectory'], "w+"), fieldnames=['timestamp', 'theta', 'thetadot', 'target_1', 'target_2'])
            self.data_files['setup'].writeheader()
            self.data_files['trajectory'].writeheader()

        # Initialize robot connection
        self.rtde_c_1 = None
        self.rtde_r_1 = None

        self.rtde_c_2 = None
        self.rtde_r_2 = None

        self.num_dof = 12
        self.init_joint_position = [1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0]

        if self.use_hardware:
            try:
                self.rtde_c_1 = RTDEControl("192.168.0.120")
                self.rtde_r_1 = RTDEReceive("192.168.0.120")

                self.rtde_c_2 = RTDEControl("192.168.0.124")
                self.rtde_r_2 = RTDEReceive("192.168.0.124")
                print("Connection with UR5e established.")
            except Exception as e:
                print(f"Could not connect to robot: {e}")
                rclpy.shutdown()
                return

            # Move to initial position
            self.move_to_start()

        setup = list(csv.DictReader(open(os.path.join(PACKAGE_DIR, 'data', 'manual', 'setup', f'setup_000.csv'), "r")))[-1]
        
        # Initialize MuJoCo model and data
        model_path = os.path.join(get_package_share_directory('real_demo'),'sampling_based_planner', 'ur5e_hande_mjx', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.1
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:self.num_dof] = self.init_joint_position

        table_1_pos = json.loads(setup['table_1'])
        table_2_pos = json.loads(setup['table_2'])

        self.model.body(name='table_1').pos = table_1_pos
        # self.model.body(name='table1_marker').pos = setup['marker1']
        self.model.body(name='table_2').pos = table_2_pos
        # self.model.body(name='table2_marker').pos = setup['marker2']
        
        # Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self.model,
            data=self.data,
            num_dof=12,
            num_batch=500,
            num_steps=15,
            maxiter_cem=1,
            maxiter_projection=5,
            w_pos=3.0,
            w_rot=0.5,
            w_col=500.0,
            num_elite=0.05,
            timestep=0.1,
            position_threshold=0.06,
            rotation_threshold=0.1,
            table_1_pos=table_1_pos,
            table_2_pos=table_2_pos
        )
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = [-3.5, 0.0, 0.8]#[0.0, 0.0, 0.8]  
        self.viewer.cam.distance = 5.0 
        self.viewer.cam.azimuth = 90.0 
        self.viewer.cam.elevation = -30.0 

        # Setup subscribers
        self.subscription_object1 = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/object1/pose',
            self.object1_callback,
            qos_profile 
        )
        self.subscription_object1 = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/object2/pose',
            self.object2_callback,
            qos_profile 
        )
        self.subscription_obstacle1 = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/obstacle1/pose',
            self.obstacle1_callback,
            qos_profile 
        )
        
        # Setup control timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.rtde_c_1.moveJ(self.init_joint_position[:self.num_dof//2], asynchronous=False)
        self.rtde_c_2.moveJ(self.init_joint_position[self.num_dof//2:], asynchronous=False)
        print("Moved to initial pose.")

    def close_connection(self):
        if self.use_hardware:
            """Cleanup robot connection"""
            if self.rtde_c_1:
                self.rtde_c_1.speedStop()
                self.rtde_c_1.disconnect()

            if self.rtde_c_2:
                self.rtde_c_2.speedStop()
                self.rtde_c_2.disconnect()
            print("Disconnected from UR5 Robot")

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
        obstacle_pos = [-pose.position.x, -pose.position.y, pose.position.z]
        obstacle_rot = [pose.orientation.w, pose.orientation.x, 
                        pose.orientation.y, pose.orientation.z]
        # self.planner.update_obstacle(obstacle_pos, obstacle_rot)

    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()

        if self.use_hardware:
            # Get current state
            current_pos_1 = np.array(self.rtde_r_1.getActualQ())
            current_vel_1 = np.array(self.rtde_r_1.getActualQd())

            current_pos_2 = np.array(self.rtde_r_2.getActualQ())
            current_vel_2 = np.array(self.rtde_r_2.getActualQd())

            current_pos = np.concatenate((current_pos_1, current_pos_2), axis=None)
            current_vel = np.concatenate((current_vel_1, current_vel_2), axis=None)
        else:
            current_pos = self.data.qpos[:self.planner.num_dof]
            current_vel = self.data.qvel[:self.planner.num_dof]
        
        
        
        # Compute control
        thetadot, cost, cost_g, cost_r, cost_c = self.planner.compute_control(
            current_pos, current_vel)
        
        if self.use_hardware:
            # Send velocity command
            self.rtde_c_1.speedJ(thetadot[:self.planner.num_dof//2], acceleration=1.4, time=0.1)
            self.rtde_c_2.speedJ(thetadot[self.planner.num_dof//2:], acceleration=1.4, time=0.1)

            # Update MuJoCo state
            current_pos = np.concatenate((np.array(self.rtde_r_1.getActualQ()), np.array(self.rtde_r_2.getActualQ())), axis=None)
            self.data.qpos[:self.planner.num_dof] = current_pos
            mujoco.mj_forward(self.model, self.data)
        else:
            self.data.qvel[:self.planner.num_dof] = thetadot
            mujoco.mj_step(self.model, self.data)

        theta = self.data.qpos[:self.planner.num_dof]

        if self.record_data_:
            self.record_data(theta=theta, thetadot=thetadot)
        
        # Update viewer
        self.viewer.sync()
        
        # Print debug info
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | '
              f'Cost g: {"%.2f"%(float(cost_g))} | '
              f'Cost c: {"%.2f"%(float(cost_c))} | '
              f'Cost: {np.round(cost, 2)}')
        
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
    mocap_listener = MocapListener()
    print("Initialized node.")
    
    try:
        rclpy.spin(mocap_listener)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        mocap_listener.close_connection()
        mocap_listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()