import numpy as np
import os
import mujoco
from collision_free_ik.mink.lie import SE3, SO3
from collision_free_ik.mink.configuration import Configuration
from collision_free_ik.mink.limits.configuration_limit import ConfigurationLimit
from collision_free_ik.mink.solve_ik import solve_ik
from collision_free_ik.mink.tasks.frame_task import FrameTask


class InverseKinematicsSolver:
    def __init__(self, model, joint_positions=None, tcp="tcp"):
        # Load MuJoCo model and data
        self.model = model 
        self.data = mujoco.MjData(self.model)

        # Set initial joint configuration
        if joint_positions is None:
            joint_positions = np.zeros(self.model.nq)
        self.configuration = Configuration(self.model, joint_positions)

        # Create IK limits
        self.limits = ConfigurationLimit(self.model)

        # Create task object (target set later)
        self.frame_task = FrameTask(
            frame_name=tcp,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            gain=1.0,
            lm_damping=0.0
        )

    def set_target(self, target_position, target_quaternion):
        """
        Set the target pose for the end-effector.
        Args:
            target_position: [x, y, z] in meters
            target_quaternion: [w, x, y, z]
        """
        rotation = SO3(wxyz=target_quaternion)
        target_pose = SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=target_position
        )
        self.frame_task.set_target(target_pose)

    def solve(self, dt=10.0, solver='daqp', damping=1e-4, safety_break=True):
        """
        Solve IK problem given the current target.
        Returns:
            new_q: New joint positions (solution)
        """
        q_vel = solve_ik(
            configuration=self.configuration,
            tasks=[self.frame_task],
            dt=dt,
            solver=solver,
            damping=damping,
            safety_break=safety_break,
            limits=[self.limits]
        )
        return q_vel
