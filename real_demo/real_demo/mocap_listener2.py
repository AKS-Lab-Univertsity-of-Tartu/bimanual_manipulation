import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import os
import time
import numpy as np
import mujoco
from mujoco import viewer

from sampling_based_planner.mpc_planner_2 import run_cem_planner
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

class MocapListener(Node):
    def __init__(self):
        super().__init__('mocap_listener')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        # Initialize robot connection
        self.rtde_c = None
        self.rtde_r = None
        self.init_joint_position = [1.5, -1.8, 1.75, -1.25, -1.6, 0.0]
        
        try:
            self.rtde_c = RTDEControl("192.168.0.120")
            self.rtde_r = RTDEReceive("192.168.0.120")
            print("Connection with UR5e established.")
        except Exception as e:
            print(f"Could not connect to robot: {e}")
            rclpy.shutdown()
            return

        # Move to initial position
        self.move_to_start()
        
        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path("path_to_your_model.xml")
        self.data = mujoco.MjData(self.model)
        
        # Initialize CEM/MPC planner
        self.planner = run_cem_planner(
            model=self.model,
            data=self.data,
            num_dof=6,
            num_batch=500,
            num_steps=20,
            maxiter_cem=1,
            maxiter_projection=5,
            w_pos=3.0,
            w_rot=0.5,
            w_col=500.0,
            num_elite=0.05,
            timestep=0.05,
            position_threshold=0.06,
            rotation_threshold=0.1
        )
        
        # Setup viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]  
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
        self.subscription_obstacle1 = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/obstacle1/pose',
            self.obstacle1_callback,
            qos_profile 
        )
        
        # Setup control timer
        self.timer = self.create_timer(0.05, self.control_loop)

    def move_to_start(self):
        """Move robot to initial joint position"""
        self.rtde_c.moveJ(self.init_joint_position, asynchronous=False)
        print("Moved to initial pose.")

    def close_connection(self):
        """Cleanup robot connection"""
        if self.rtde_c:
            self.rtde_c.speedStop()
            self.rtde_c.disconnect()
        print("Disconnected from UR5 Robot")

    def object1_callback(self, msg):
        """Callback for target object pose updates"""
        pose = msg.pose
        target_pos = [-pose.position.x, -pose.position.y, pose.position.z]
        target_rot = [pose.orientation.w, pose.orientation.x, 
                      pose.orientation.y, pose.orientation.z]
        self.planner.update_target(target_pos, target_rot)

    def obstacle1_callback(self, msg):
        """Callback for obstacle pose updates"""
        pose = msg.pose
        obstacle_pos = [-pose.position.x, -pose.position.y, pose.position.z]
        obstacle_rot = [pose.orientation.w, pose.orientation.x, 
                        pose.orientation.y, pose.orientation.z]
        self.planner.update_obstacle(obstacle_pos, obstacle_rot)

    def control_loop(self):
        """Main control loop running at fixed interval"""
        start_time = time.time()
        
        # Get current state
        current_pos = np.array(self.rtde_r.getActualQ())
        current_vel = np.array(self.rtde_r.getActualQd())
        
        # Update MuJoCo state
        self.data.qpos[:self.planner.num_dof] = current_pos
        mujoco.mj_forward(self.model, self.data)
        
        # Compute control
        thetadot, cost, cost_g, cost_c = self.planner.compute_control(
            current_pos, current_vel)
        
        # Send velocity command
        self.rtde_c.speedJ(thetadot, acceleration=1.4, time=0.05)
        
        # Update viewer
        self.viewer.sync()
        
        # Print debug info
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | '
              f'Cost g: {"%.2f"%(float(cost_g))} | '
              f'Cost c: {"%.2f"%(float(cost_c))} | '
              f'Cost: {np.round(cost, 2)}')

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