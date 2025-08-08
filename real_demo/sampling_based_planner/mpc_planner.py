from sampling_based_planner.mjx_planner import cem_planner
from sampling_based_planner.quat_math import quaternion_distance, quaternion_multiply, rotation_quaternion
from sampling_based_planner.Simple_MLP.mlp_singledof import MLP, MLPProjectionFilter
from ik_based_planner.ik_solver import InverseKinematicsSolver

import mujoco
from mujoco import viewer
import jax.numpy as jnp
import jax

import numpy as np
import torch 
import contextlib
from io import StringIO


class run_cem_planner:
    def __init__(self, model, data, num_dof=12, num_batch=500, num_steps=20, 
                 maxiter_cem=1, maxiter_projection=5, num_elite=0.05, timestep=0.05,
                 position_threshold=0.06, rotation_threshold=0.1,
                 ik_pos_thresh=0.08, ik_rot_thresh=0.1, 
                 collision_free_ik_dt=2.0, inference=False, rnn=None,
                 max_joint_pos=180.0*np.pi/180.0, max_joint_vel=1.0, 
                 max_joint_acc=2.0, max_joint_jerk=4.0,
                 device='cuda', cost_weights=None):
        
        # Initialize parameters
        self.model = model
        self.data = data
        self.num_dof = num_dof
        self.num_batch = num_batch
        self.num_steps = num_steps
        self.maxiter_cem = maxiter_cem
        self.maxiter_projection = maxiter_projection
        self.num_elite = num_elite
        self.timestep = timestep
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.ik_pos_thresh = ik_pos_thresh if ik_pos_thresh else 1.1 * position_threshold
        self.ik_rot_thresh = ik_rot_thresh if ik_rot_thresh else 1.1 * rotation_threshold
        self.collision_free_ik_dt = collision_free_ik_dt
        self.inference = inference
        self.device = device

        self.cost_weights = cost_weights

        # Initialize CEM planner
        self.cem = cem_planner(
            model=model,
            num_dof=num_dof, 
            num_batch=num_batch, 
            num_steps=num_steps, 
            maxiter_cem=maxiter_cem,
            num_elite=num_elite,
            timestep=timestep,
            maxiter_projection=maxiter_projection,
            max_joint_pos=max_joint_pos,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            max_joint_jerk=max_joint_jerk,

        )
        
        # Initialize CEM variables
        self.xi_mean_single = jnp.zeros(self.cem.nvar_single)
        self.xi_cov_single = 10*jnp.identity(self.cem.nvar_single)
        self.xi_mean = jnp.tile(self.xi_mean_single, self.cem.num_dof)
        self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), self.xi_cov_single)
        self.lamda_init = jnp.zeros((num_batch, self.cem.nvar))
        self.s_init = jnp.zeros((num_batch, self.cem.num_total_constraints))
        self.key = jax.random.PRNGKey(0)
        
        # Get references for both arms
        self.target_pos_0 = data.xpos[model.body(name="target_0").id]
        self.target_rot_0 = data.xquat[model.body(name="target_0").id].copy()
        self.target_0 = np.concatenate([self.target_pos_0, self.target_rot_0])

        self.target_pos_2 = data.xpos[model.body(name="ball").id]
        self.target_rot_2 = data.xquat[model.body(name="ball").id].copy()
        self.target_2 = np.concatenate([self.target_pos_2, self.target_rot_2])

        # Get obstacle reference
        self.obstacle_pos = data.mocap_pos[model.body_mocapid[model.body(name='obstacle').id]]
        self.obstacle_rot = data.mocap_quat[model.body_mocapid[model.body(name='obstacle').id]]
        
        # Get TCP references for both arms
        self.tcp_id_0 = model.site(name="tcp_0").id
        self.hande_id_0 = model.body(name="hande_0").id
        self.tcp_id_1 = model.site(name="tcp_1").id
        self.hande_id_1 = model.body(name="hande_1").id
        
        # Initialize MLP if inference is enabled
        if inference:
            self.mlp_model = self._load_mlp_projection_model(num_steps + 1 + 1, rnn, maxiter_projection)

    def _load_mlp_projection_model(self, num_feature, rnn_type, maxiter_projection):
        """Load the MLP projection model for inference"""
        enc_inp_dim = num_feature
        mlp_inp_dim = enc_inp_dim
        hidden_dim = 1024
        mlp_out_dim = 2 * self.cem.nvar_single + self.cem.num_total_constraints_per_dof

        mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)
        with contextlib.redirect_stdout(StringIO()):
            model = MLPProjectionFilter(
                mlp=mlp,
                num_batch=self.num_batch,
                num_dof=self.num_dof,
                num_steps=self.num_steps,
                timestep=self.timestep,
                v_max=self.cem.v_max,
                a_max=self.cem.a_max,
                j_max=self.cem.j_max,
                p_max=self.cem.p_max,
                maxiter_projection=maxiter_projection,
            ).to(self.device)

            weight_path = './training_weights/mlp_learned_single_dof.pth'
            model.load_state_dict(torch.load(weight_path, weights_only=True))
            model.eval()
        
        return model

    def _robust_scale(self, input_nn):
        """Normalize input using median and IQR"""
        inp_median_ = torch.median(input_nn, dim=0).values
        inp_q1 = torch.quantile(input_nn, 0.25, dim=0)
        inp_q3 = torch.quantile(input_nn, 0.75, dim=0)
        inp_iqr_ = inp_q3 - inp_q1
        
        # Handle constant features
        inp_iqr_ = torch.where(inp_iqr_ == 0, 1.0, inp_iqr_)
        
        return (input_nn - inp_median_) / inp_iqr_

    def _append_torch_tensors(self, variable_single_dof, variable_multi_dof):
        """Helper function to append tensors"""
        if isinstance(variable_multi_dof, list):
            if len(variable_multi_dof) == 0:
                return variable_single_dof
            variable_multi_dof = torch.stack(variable_multi_dof, dim=0)
        
        return torch.cat([variable_multi_dof, variable_single_dof], dim=1)

    def update_targets(self, target_idx=0, target_pos=None, target_rot=None):
        """Update target positions and rotations for both arms"""
        if target_idx==0:
            self.target_pos_0 = target_pos
            self.target_rot_0 = target_rot
            self.target_0 = np.concatenate([self.target_pos_0, self.target_rot_0])
        elif target_idx==1:
            self.target_pos_1 = target_pos
            self.target_rot_1 = target_rot
            self.target_1 = np.concatenate([self.target_pos_1, self.target_rot_1])
        
    def update_obstacle(self, obstacle_pos, obstacle_rot):
        """Update obstacle position and rotation"""
        self.obstacle_pos = obstacle_pos
        self.obstacle_rot = obstacle_rot
        
    def compute_control(self, current_pos, current_vel, task):
        """Compute optimal control using CEM/MPC for dual-arm system"""
        
        # Handle covariance matrix numerical stability
        if np.isnan(self.xi_cov).any():
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), 10*jnp.identity(self.cem.nvar_single))
        if np.isnan(self.xi_mean).any():
            self.xi_mean = jnp.zeros(self.cem.nvar)

        try:
            np.linalg.cholesky(self.xi_cov)
        except np.linalg.LinAlgError:
            self.xi_cov = jnp.kron(jnp.eye(self.cem.num_dof), 10*jnp.identity(self.cem.nvar_single))  
        
        # Generate samples
        self.xi_samples, self.key = self.cem.compute_xi_samples(self.key, self.xi_mean, self.xi_cov)
        xi_samples_reshaped = self.xi_samples.reshape(self.num_batch, self.cem.num_dof, self.cem.nvar_single)

        # MLP inference if enabled
        if self.inference:
            xi_projected_nn_output = []
            lamda_init_nn_output = []
            s_init_nn_output = []
            
            for i in range(self.cem.joint_mask_pos):
                if self.cem.joint_mask_pos[i]:
                    theta_init = np.tile(self.data.qpos[i], (self.num_batch, 1))
                    v_start = np.tile(self.data.qvel[i], (self.num_batch, 1))
                    xi_samples_single = xi_samples_reshaped[:, i, :]
                    inp = np.hstack([xi_samples_single, theta_init, v_start])
                    inp_torch = torch.tensor(inp).float().to(self.device)
                    inp_norm_torch = self._robust_scale(inp_torch)
                    neural_output_batch = self.mlp_model.mlp(inp_norm_torch)
                    
                    xi_projected_nn_output_single = neural_output_batch[:, :self.cem.nvar_single]
                    lamda_init_nn_output_single = neural_output_batch[:, self.cem.nvar_single: 2*self.cem.nvar_single]
                    s_init_nn_output_single = neural_output_batch[:, 2*self.cem.nvar_single: 2*self.cem.nvar_single + self.cem.num_total_constraints_per_dof]
                    s_init_nn_output_single = torch.maximum(
                        torch.zeros((self.num_batch, self.cem.num_total_constraints_per_dof), device=self.device), 
                        s_init_nn_output_single)
                    
                    xi_projected_nn_output = self._append_torch_tensors(
                        xi_projected_nn_output_single, xi_projected_nn_output)
                    lamda_init_nn_output = self._append_torch_tensors(
                        lamda_init_nn_output_single, lamda_init_nn_output)
                    s_init_nn_output = self._append_torch_tensors(
                        s_init_nn_output_single, s_init_nn_output)
            
            self.lamda_init = np.array(lamda_init_nn_output.cpu().detach().numpy())
            self.s_init = np.array(s_init_nn_output.cpu().detach().numpy())

        # if task == "pick":
        #     self.cost_weights['pick'] = 1
        #     self.cost_weights['move'] = 0
        # elif task == 'move':
        #     self.cost_weights['pick'] = 0
        #     self.cost_weights['move'] = 1

        # CEM computation
        cost, best_cost_list, thetadot_horizon, theta_horizon, \
        self.xi_mean, self.xi_cov, thd_all, th_all, avg_primal_res, avg_fixed_res, \
        primal_res, fixed_res, idx_min, ball_out = self.cem.compute_cem(
            self.xi_mean,
            self.xi_cov,
            current_pos,
            current_vel,
            np.zeros(self.num_dof),  # Zero initial acceleration
            self.target_0,
            self.target_2,
            self.lamda_init,
            self.s_init,
            self.xi_samples,
            self.cost_weights,
        )

        # Get mean velocity command (average middle 90% of trajectory)
        thetadot_cem = np.mean(thetadot_horizon[1:int(self.num_steps*0.3)], axis=0)

        thetadot_0 = thetadot_cem[:6]
        thetadot_1 = thetadot_cem[6:]
        
        # print("BALL INIT:", self.target_2[:3], flush=True)
        # print("BALL:", ball_out[:, :3], flush=True)

        # if task == "pick":

        #     # Check if we should switch to collision-free IK for each arm
        #     current_cost_g_0 = np.linalg.norm(self.data.site_xpos[self.tcp_id_0] - self.target_pos_0)
        #     current_cost_r_0 = quaternion_distance(self.data.xquat[self.hande_id_0], self.target_rot_0)
                
        #     current_cost_g_1 = np.linalg.norm(self.data.site_xpos[self.tcp_id_1] - self.target_pos_1)
        #     current_cost_r_1 = quaternion_distance(self.data.xquat[self.hande_id_1], self.target_rot_1)
            
        #     joint_states = np.zeros(self.cem.joint_mask_pos.shape)
        #     joint_states[self.cem.joint_mask_pos] = current_pos

        #     # Arm 0 control
        #     if current_cost_g_0 < self.ik_pos_thresh and current_cost_r_0 < self.ik_rot_thresh:
        #         print("Activated ik solution for arm 0")
        #         ik_solver_0 = InverseKinematicsSolver(
        #             self.model, joint_states, "tcp_0")
        #         ik_solver_0.set_target(self.target_pos_0, self.target_rot_0)
        #         thetadot_0 = ik_solver_0.solve(dt=self.collision_free_ik_dt)[self.cem.joint_mask_vel][:self.num_dof//2]
            
        #     # Arm 1 control
        #     if current_cost_g_1 < self.ik_pos_thresh and current_cost_r_1 < self.ik_rot_thresh:
        #         print("Activated ik solution for arm 1")
        #         ik_solver_1 = InverseKinematicsSolver(
        #             self.model, joint_states, "tcp_1")
        #         ik_solver_1.set_target(self.target_pos_1, self.target_rot_1)
        #         thetadot_1 = ik_solver_1.solve(dt=self.collision_free_ik_dt)[self.cem.joint_mask_vel][self.num_dof//2:]

        # Combine control commands
        thetadot = np.concatenate((thetadot_0, thetadot_1))
        
        return thetadot, cost, best_cost_list, thetadot_horizon, theta_horizon