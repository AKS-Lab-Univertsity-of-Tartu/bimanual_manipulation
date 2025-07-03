import numpy as np
from mjx_planner import cem_planner
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
import os
from mujoco import viewer
import matplotlib.pyplot as plt
from quat_math import rotation_quaternion, quaternion_multiply, quaternion_distance
import argparse
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from functools import partial
from Simple_MLP.mlp_singledof import MLP, MLPProjectionFilter
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import contextlib
from io import StringIO

import sys
#Enable python to search for modules in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ik_based_planner.ik_solver import InverseKinematicsSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class MuJoCoPointCloudGenerator:
    def __init__(self, model, cam_name="camera1", height=480, width=640, output_dir="pcd_data_simulation"):
        self.model = model
        self.cam_name = cam_name
        self.height = height
        self.width = width
        self.output_dir = output_dir
        
        # Get camera ID
        try:
            self.cam_id = model.camera(cam_name).id
        except:
            raise ValueError(f"Camera '{cam_name}' not found in model")
        
        # Initialize renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Calculate camera intrinsics
        fovy = model.cam_fovy[self.cam_id]
        self.f = height / (2 * np.tan(np.deg2rad(fovy / 2)))
        self.cx, self.cy = width / 2, height / 2
        
        # Create meshgrids for projection
        self.i, self.j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Accumulation buffers
        self.accumulated_points = np.empty((0, 3))
        self.accumulated_colors = np.empty((0, 3))
        
        print(f"Initialized point cloud generator with camera '{cam_name}'")

    def generate_point_cloud(self, data, max_depth=10.0, downsample_factor=1):
        """Generate point cloud from current scene"""
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.cam_name)

        # Render RGB and depth
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()

        assert depth.shape[:2] == rgb.shape[:2], \
            f"Mismatch: depth.shape={depth.shape}, rgb.shape={rgb.shape}"

        H, W = depth.shape
        z = depth

        if downsample_factor > 1:
            z = z[::downsample_factor, ::downsample_factor]
            rgb = rgb[::downsample_factor, ::downsample_factor, :]

        H, W = z.shape

        # Generate pixel grid
        i, j = np.meshgrid(np.arange(W), np.arange(H))
        i = i.astype(np.float32)
        j = j.astype(np.float32)

        x = (i - self.cx) * z / self.f
        y = (j - self.cy) * z / self.f

        points_cam = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)

        # Transform to world coordinates
        cam_pos = data.cam_xpos[self.cam_id]
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3)
        points_world = (cam_mat @ points_cam.T).T + cam_pos

        # Flatten RGB
        rgb_flat = rgb.reshape(-1, 3)

        # Filter valid points
        valid_mask = ~(np.isnan(points_world).any(axis=1) | np.isinf(points_world).any(axis=1))
        valid_mask &= np.abs(points_world[:, 2]) < max_depth
        valid_mask &= z.flatten() > 0.01

        points = points_world[valid_mask]
        colors = rgb_flat[valid_mask].astype(np.uint8)

        return points, colors


    def accumulate_point_cloud(self, data, max_depth=10.0, downsample_factor=1):
        """Generate and accumulate point cloud"""
        points, colors = self.generate_point_cloud(data, max_depth, downsample_factor)
        
        # Append to accumulated buffers
        self.accumulated_points = np.vstack((self.accumulated_points, points))
        self.accumulated_colors = np.vstack((self.accumulated_colors, colors))
        
        return points, colors

    def deduplicate_points(self, points, colors, threshold=0.01):
        """Merge nearby points"""
        if len(points) == 0:
            return points, colors
            
        nbrs = NearestNeighbors(radius=threshold).fit(points)
        clusters = nbrs.radius_neighbors(points, return_distance=False)
        
        unique_points = []
        unique_colors = []
        visited = set()
        
        for i, cluster in enumerate(clusters):
            if i not in visited:
                cluster_points = points[cluster]
                cluster_colors = colors[cluster]
                unique_points.append(np.mean(cluster_points, axis=0))
                unique_colors.append(np.mean(cluster_colors, axis=0))
                visited.update(cluster)
                
        return np.array(unique_points), np.array(unique_colors)

    def save_accumulated_pcd(self, filename="accumulated.pcd", deduplicate=True):
        """Save all accumulated points as a single PCD file"""
        if len(self.accumulated_points) == 0:
            print("Warning: No points accumulated")
            return None
            
        points = self.accumulated_points
        colors = self.accumulated_colors
        
        if deduplicate:
            points, colors = self.deduplicate_points(points, colors)
            
        output_path = os.path.join(self.output_dir, filename)
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.io.write_point_cloud(output_path, pcd_o3d)
        
        print(f"Saved accumulated point cloud with {len(points)} points to {output_path}")
        return output_path

    def reset_accumulation(self):
        """Clear accumulated points"""
        self.accumulated_points = np.empty((0, 3))
        self.accumulated_colors = np.empty((0, 3))

def robust_scale(input_nn: torch.Tensor) -> torch.Tensor:
    """Normalize input using median and IQR"""
    inp_median_ = torch.median(input_nn, dim=0).values
    inp_q1 = torch.quantile(input_nn, 0.25, dim=0)
    inp_q3 = torch.quantile(input_nn, 0.75, dim=0)
    inp_iqr_ = inp_q3 - inp_q1
    
    # Handle constant features
    inp_iqr_ = torch.where(inp_iqr_ == 0, 1.0, inp_iqr_)
    
    inp_norm = (input_nn - inp_median_) / inp_iqr_
    return inp_norm

@partial(jax.jit, static_argnames=['nvar', 'num_batch'])
def compute_xi_samples(key, xi_mean, xi_cov, nvar, num_batch):
    key, subkey = jax.random.split(key)
    xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(nvar), (num_batch, ))
    return xi_samples, key

def load_mlp_projection_model(num_feature, rnn_type, cem, maxiter_projection, device='cuda'):
    enc_inp_dim = num_feature
    mlp_inp_dim = enc_inp_dim
    hidden_dim = 1024
    mlp_out_dim = 2 * cem.nvar_single + cem.num_total_constraints_per_dof

    mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)
    with contextlib.redirect_stdout(StringIO()):
        model = MLPProjectionFilter(
            mlp=mlp,
            num_batch=cem.num_batch,
            num_dof=cem.num_dof,
            num_steps=cem.num,
            timestep=cem.t,
            v_max=cem.v_max,
            a_max=cem.a_max,
            j_max=cem.j_max,
            p_max=cem.p_max,
            maxiter_projection=maxiter_projection,
        ).to(device)

        weight_path = f'./training_weights/mlp_learned_single_dof.pth'
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
    
    return model

def append_torch_tensors(variable_single_dof, variable_multi_dof):
    if isinstance(variable_multi_dof, list):
        if len(variable_multi_dof) == 0:
            return variable_single_dof
        else:
            variable_multi_dof = torch.stack(variable_multi_dof, dim=0)
    
    variable_multi_dof = torch.cat([variable_multi_dof, variable_single_dof], dim=1)
    return variable_multi_dof

def run_cem_planner(
    num_dof=None,
    num_batch=None,
    num_steps=None,
    maxiter_cem=None,
    maxiter_projection=None,
    w_pos=None,
    w_rot=None,
    w_col=None,
    num_elite=None,
    timestep=None,
    initial_qpos=None,
    ik_pos_thresh=None,
    ik_rot_thresh=None,
    collision_free_ik_dt=None,
    target_names=None,
    show_viewer=None,
    cam_distance=None,
    show_contact_points=None,
    position_threshold=None,
    rotation_threshold=None,
    save_data=None,
    data_dir=None,
    stop_at_final_target=None,
    inference=None,
    rnn=None,
    max_joint_pos=None,
    max_joint_vel=None,
    max_joint_acc=None,
    max_joint_jerk=None,
    generate_pcd=None,
    accumulate_pcd=None,
    pcd_interval=None,
    cam_name=None,
):
    # Initialize data structures
    index_list = []
    cost_g_list = []
    cost_r_list = []
    cost_c_list = []
    cost_list = []
    thetadot_list = []
    theta_list = []
    best_vel_list = []
    avg_primal_residual_list = []
    avg_fixed_point_residual_list = []
    best_cost_primal_residual_list = []
    best_cost_fixed_point_residual_list = []
    target_pos_list = []
    target_quat_list = []

    # Initialize CEM planner
    cem = cem_planner(
        num_dof=num_dof, 
        num_batch=num_batch, 
        num_steps=num_steps, 
        maxiter_cem=maxiter_cem,
        w_pos=w_pos,
        w_rot=w_rot,
        w_col=w_col,
        num_elite=num_elite,
        timestep=timestep,
        maxiter_projection=maxiter_projection,
        max_joint_pos=max_joint_pos,
        max_joint_vel=max_joint_vel,
        max_joint_acc=max_joint_acc,
        max_joint_jerk=max_joint_jerk
    )

    model = cem.model
    data = cem.data

    
    
    # Defining Obstacle position here is not needed as that is taken from environment
    # For mujoco (official Python bindings):
    obstacle_indices = [i for i in range(cem.model.nbody) 
                    if cem.model.body(i).name.startswith("obstacle_")]
    obst_pos = [cem.mjx_data.xpos[i] for i in obstacle_indices]
    obst_quat = [cem.mjx_data.xquat[i] for i in obstacle_indices]

    data.qpos[:num_dof] = jnp.array(initial_qpos)
    mujoco.mj_forward(model, data)

    # Initialize point cloud generator if enabled
    if generate_pcd:
        pcd_gen = MuJoCoPointCloudGenerator(model=model, cam_name=cam_name, output_dir=data_dir)

    # Initialize CEM variables
    xi_mean_single = jnp.zeros(cem.nvar_single)
    xi_cov_single = 10*jnp.identity(cem.nvar_single)
    xi_mean_init = jnp.tile(xi_mean_single, cem.num_dof)
    xi_cov_init = jnp.kron(jnp.eye(cem.num_dof), xi_cov_single)
    xi_mean = xi_mean_init
    xi_cov = xi_cov_init
    xi_samples, key = cem.compute_xi_samples(cem.key, xi_mean, xi_cov)
    lamda_init = jnp.zeros((cem.num_batch, cem.nvar))
    s_init = jnp.zeros((cem.num_batch, cem.num_total_constraints))

    #Initialize EE pose
    init_position_1 = data.site_xpos[model.site(name="tcp").id].copy()
    init_rotation_1 = data.xquat[model.body(name="hande").id].copy()

    init_position_2 = data.site_xpos[model.site(name="tcp_2").id].copy()
    init_rotation_2 = data.xquat[model.body(name="hande_2").id].copy()

    # Load MLP model if inference is enabled
    if inference:
        mlp_model = load_mlp_projection_model(
            num_steps + 1 + 1, rnn, cem, maxiter_projection, device=device)

    timestep_counter = 0
    target_idx = 0
    current_target = target_names[target_idx]
    target_reached = False #Initialize

    target_positions_1 = [
        [-0.3, 0.3, 0.3],
        [-0.1, -0.35, 0.6],
        [-0.3, -0.1, 0.3],
        init_position_1
    ]

    target_rotations_1 = [
        rotation_quaternion(-135, np.array([1,0,0])),
        quaternion_multiply(rotation_quaternion(90, np.array([0,0,1])),rotation_quaternion(135, np.array([1,0,0]))),
        quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])),rotation_quaternion(-90, np.array([0,1,0]))),
        init_rotation_1
    ]

    target_positions_2 = [
        [-0.3, -0.3, 0.3],
        [-0.3, 0.4, 0.6],
        [-0.3, 0.1, 0.3],
        init_position_2
    ]

    target_rotations_2 = [
        quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])), rotation_quaternion(135, np.array([1,0,0]))),
        quaternion_multiply(rotation_quaternion(-90, np.array([0,0,1])),rotation_quaternion(-135, np.array([1,0,0]))),
        quaternion_multiply(rotation_quaternion(0, np.array([0,0,1])),rotation_quaternion(90, np.array([0,1,0]))),
        init_rotation_2
    ]

    target_idx = 0

    target_pos_1 = model.body(name="target_0").pos
    target_quat_1 = model.body(name="target_0").quat
    target_pos_2 = model.body(name="target_1").pos
    target_quat_2 = model.body(name="target_1").quat



    if show_viewer:
        with viewer.launch_passive(model, data) as viewer_:
            viewer_.cam.distance = cam_distance
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            
            try:
                while viewer_.is_running():
                    timestep_counter += 1
                    start_time = time.time()
                    
                    # Point cloud generation
                    if generate_pcd and timestep_counter % pcd_interval == 0:
                        if accumulate_pcd:
                            pcd_gen.accumulate_point_cloud(data, downsample_factor=2)
                        else:
                            points, colors = pcd_gen.generate_point_cloud(data)
                            filename = f"pcd_frame_{timestep_counter:04d}.pcd"
                            pcd_gen.save_pcd_binary(points, colors, filename)

                    # Main CEM planning loop
                    # target_pos = model.body(name=current_target).pos if current_target != "home" else data.site_xpos[model.site(name="tcp").id].copy()
                    # target_quat = model.body(name=current_target).quat if current_target != "home" else data.xquat[model.body(name="hande").id].copy()
                    

                    # Determine target position and orientation
                    # if current_target != "home":
                    #     target_pos = model.body(name=current_target).pos
                    #     target_quat = model.body(name=current_target).quat
                    # else:
                    #     target_pos = init_position
                    #     target_quat = init_quaternion

                    if np.isnan(xi_cov).any():
                        xi_cov = xi_cov_init
                    if np.isnan(xi_mean).any():
                        xi_mean = xi_mean_init

                    try:
                        np.linalg.cholesky(xi_cov)
                    except np.linalg.LinAlgError:
                        xi_cov = xi_cov_init    

                    xi_samples, key = cem.compute_xi_samples(cem.key, xi_mean, xi_cov)
                    xi_samples_reshaped = xi_samples.reshape(cem.num_batch, cem.num_dof, cem.nvar_single)

                    if inference:
                        xi_projected_nn_output = []
                        lamda_init_nn_output = []
                        s_init_nn_output = []
                        
                        for i in range(cem.num_dof):
                            theta_init = np.tile(data.qpos[i], (num_batch,1))
                            v_start = np.tile(data.qvel[i], (num_batch,1)) 
                            xi_samples_single = xi_samples_reshaped[:, i, :]
                            inp = np.hstack([xi_samples_single, theta_init, v_start])
                            inp_torch = torch.tensor(inp).float().to(device)
                            inp_norm_torch = robust_scale(inp_torch)
                            neural_output_batch = mlp_model.mlp(inp_norm_torch)
                            
                            xi_projected_nn_output_single = neural_output_batch[:, :cem.nvar_single]
                            lamda_init_nn_output_single = neural_output_batch[:, cem.nvar_single: 2*cem.nvar_single]
                            s_init_nn_output_single = neural_output_batch[:, 2*cem.nvar_single: 2*cem.nvar_single + cem.num_total_constraints_per_dof]
                            s_init_nn_output_single = torch.maximum(torch.zeros((cem.num_batch, cem.num_total_constraints_per_dof), device=device), s_init_nn_output_single)
                            
                            xi_projected_nn_output = append_torch_tensors(xi_projected_nn_output_single, xi_projected_nn_output)
                            lamda_init_nn_output = append_torch_tensors(lamda_init_nn_output_single, lamda_init_nn_output)
                            s_init_nn_output = append_torch_tensors(s_init_nn_output_single, s_init_nn_output)
                        
                        lamda_init = np.array(lamda_init_nn_output.cpu().detach().numpy())
                        s_init = np.array(s_init_nn_output.cpu().detach().numpy())

                    # CEM computation
                    cost, best_cost_g, best_cost_r, best_cost_c, best_vels, best_traj, \
                    xi_mean, xi_cov, thd_all, th_all, avg_primal_res, avg_fixed_res, \
                    primal_res, fixed_res, idx_min = cem.compute_cem(
                        xi_mean,
                        xi_cov,
                        data.qpos[:num_dof],
                        data.qvel[:num_dof],
                        data.qacc[:num_dof],
                        target_pos_1,
                        target_quat_1,
                        target_pos_2,
                        target_quat_2,
                        lamda_init,
                        s_init,
                        xi_samples
                    )

                    # Check target convergence
                    # current_cost_g = np.linalg.norm(data.site_xpos[cem.tcp_id] - target_pos)   
                    # current_cost_r = quaternion_distance(data.xquat[cem.hande_id], target_quat)
                    # current_cost = np.round(cost, 2)

                    current_cost_g = (np.linalg.norm(data.site_xpos[cem.tcp_id] - target_pos_1) + np.linalg.norm(data.site_xpos[cem.tcp_id_2] - target_pos_2))/2  
                    current_cost_r = (quaternion_distance(data.xquat[cem.hande_id], target_quat_1) + quaternion_distance(data.xquat[cem.hande_id_2], target_quat_2))/2    
                    current_cost = np.round(cost, 2)
                    
                    if current_cost_g < position_threshold and current_cost_r < rotation_threshold:
                        target_reached = True
                    else:
                        target_reached = False

                    if target_reached:
                        if target_idx == len(target_positions_1) - 1: #At last target
                            if stop_at_final_target:
                                data.qvel[:num_dof] = np.zeros(num_dof)
                            # else:
                            #     target_idx = -1
                                # current_target = target_names[target_idx]
                            
                        else:
                            target_idx += 1
                            
                            model.body(name="target_0").pos = target_positions_1[target_idx]
                            model.body(name="target_0").quat = target_rotations_1[target_idx]
                            model.body(name="target_1").pos = target_positions_2[target_idx]
                            model.body(name="target_1").quat = target_rotations_2[target_idx]
                            # current_target = target_names[target_idx]
                    

                    #ACtivate  collision free IK if cost position/rotation is less than ik_threshold
                    if current_cost_g < ik_pos_thresh and current_cost_r < ik_rot_thresh:
                        collision_free_ik = False
                    else:
                        collision_free_ik = False

                    if collision_free_ik:
                        #Collision Free IK
                        ik_solver = InverseKinematicsSolver(cem.model, data.qpos[:num_dof])

                        ik_solver.set_target(target_pos_1, target_quat_1)

                        print("\n" + "-" * 10)
                        print(">>> COLLISION-FREE IK IS ACTIVATED <<<")
                        print("-" * 10 + "\n")
                        
                        # Apply control as per MPC coupled with  CEM
                        thetadot = ik_solver.solve(dt=collision_free_ik_dt)

                    else:    

                        # Apply control as per MPC coupled with  CEM
                        thetadot = np.mean(best_vels[1:int(num_steps*0.9)], axis=0)
                    
                    data.qvel[:num_dof] = thetadot

                    # Step the simulation
                    mujoco.mj_step(model, data)

                        

                    # Store data
                    index_list.append(timestep_counter)
                    cost_g_list.append(best_cost_g)
                    cost_r_list.append(best_cost_r)
                    cost_c_list.append(best_cost_c)
                    thetadot_list.append(thetadot)
                    theta_list.append(data.qpos[:num_dof].copy())
                    cost_list.append(current_cost[-1] if isinstance(current_cost, np.ndarray) else current_cost)
                    best_vel_list.append(best_vels)
                    avg_primal_residual_list.append(np.mean(avg_primal_res, axis=1))
                    avg_fixed_point_residual_list.append(np.mean(avg_fixed_res, axis=1))
                    best_cost_primal_residual_list.append(avg_primal_res[:, idx_min])
                    best_cost_fixed_point_residual_list.append(avg_fixed_res[:, idx_min])
                    
                    #if not any(np.allclose(pos_, target_pos) for pos_ in target_pos_list):
                    target_pos_list.append(target_pos_1.copy())
                    #if not any(np.allclose(quat_, target_quat) for quat_ in target_quat_list):
                    target_quat_list.append(target_quat_1.copy())
                    
                    
                    # Print status

                    print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(current_cost_g))}'
                        f' | Cost r: {"%.2f"%(float(current_cost_r))} | Cost c: {"%.2f"%(float(best_cost_c))} | Cost: {current_cost}')
                    # print(f'eef_quat: {data.xquat[cem.hande_id]}')
                    # print(f'eef_pos', data.site_xpos[cem.tcp_id])
                    print(f'target: {current_target}')
                    # print(f'target_pos', target_pos)
                    print(f'timetstep_counter:{timestep_counter}')
                    print(f'target_reached: {target_reached}')

                    # Update viewer
                    viewer_.sync()
                    time_until_next_step = model.opt.timestep - (time.time() - start_time)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

            except KeyboardInterrupt:
                print("Interrupted by user!")
            
            finally:
                
                # Save Motion data
                if save_data:
                    print("Saving Motion, Target and Obstacle data ...")
                    os.makedirs(data_dir, exist_ok=True)

                    #Saving Motion data
                    print("Saving Motion data...")
                    np.savetxt(f'{data_dir}/index.csv', index_list, delimiter=",")
                    np.savetxt(f'{data_dir}/costs.csv', cost_list, delimiter=",")
                    np.savetxt(f'{data_dir}/thetadot.csv', thetadot_list, delimiter=",")
                    np.savetxt(f'{data_dir}/theta.csv', theta_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_g.csv', cost_g_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_r.csv', cost_r_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_c.csv', cost_c_list, delimiter=",")
                    np.savetxt(f'{data_dir}/avg_primal_residual.csv', avg_primal_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/avg_fixed_point_residual.csv', avg_fixed_point_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/best_cost_primal_residual.csv', best_cost_primal_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/best_cost_fixed_point_residual.csv', best_cost_fixed_point_residual_list, delimiter=",")
                    np.save(f'{data_dir}/best_vels.npy', np.array(best_vel_list))
                    print("Motion data saved!")
                    
                    # Save Target positions and orientations
                    print("Saving Target positions and orientations...")
                    np.savetxt(f'{data_dir}/target_positions.csv', target_pos_list, delimiter=",")
                    np.savetxt(f'{data_dir}/target_quaternions.csv', target_quat_list, delimiter=",")
                    print("Target positions and orientations saved!")
                    # Save Obstacle positions and orientations
                    print("Saving Obstacle positions and orientations...")
                    np.savetxt(f'{data_dir}/obstacle_positions.csv', obst_pos, delimiter=",")
                    np.savetxt(f'{data_dir}/obstacle_quaternions.csv', obst_quat, delimiter=",")
                    print("Obstacle positions and orientations saved!")
                    print(f"Motion, Target and Obstacle data saved to {data_dir}")

                # Save accumulated point cloud
                if generate_pcd and accumulate_pcd:
                    print("Saving accumulated point cloud...")
                    pcd_gen.save_accumulated_pcd()    

    return {
        'cost_g': cost_g_list,
        'cost_r': cost_r_list,
        'cost_c': cost_c_list,
        'cost': cost_list,
        'thetadot': thetadot_list,
        'theta': theta_list,
        'best_vels': best_vel_list,
        'primal_residual': avg_primal_residual_list,
        'fixed_point_residual': avg_fixed_point_residual_list,
        'best_cost_primal_residual': best_cost_primal_residual_list,
        'best_cost_fixed_point_residual': best_cost_fixed_point_residual_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CEM planner with configurable parameters')
    
    # CEM planner parameters
    parser.add_argument('--num_dof', type=int, default=6)
    parser.add_argument('--num_batch', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--maxiter_cem', type=int, default=1)
    parser.add_argument('--w_pos', type=float, default=20.0)
    parser.add_argument('--w_rot', type=float, default=3.0)
    parser.add_argument('--w_col', type=float, default=10.0)
    parser.add_argument('--num_elite', type=float, default=0.05)
    parser.add_argument('--timestep', type=float, default=0.05)
    parser.add_argument('--maxiter_projection', type=int, default=5)
    
    # Initial configuration
    parser.add_argument('--initial_qpos', type=float, nargs='+', default=[1.5, -1.8, 1.75, -1.25, -1.6, 0])
    
    # Visualization options
    parser.add_argument('--no_viewer', action='store_true')
    parser.add_argument('--cam_distance', type=float, default=4)
    parser.add_argument('--no_contact_points', action='store_true')
    
    # Convergence criteria
    parser.add_argument('--position_threshold', type=float, default=0.04)
    parser.add_argument('--rotation_threshold', type=float, default=0.3)
    
    # Target sequence
    parser.add_argument('--targets', type=str, nargs='+', default=["target_0", "target_1", "home"])
    
    # Save data
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--continue_after_final', action='store_true')
    
    # MLP parameters
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--rnn', type=str, default=None)
    parser.add_argument('--max_joint_pos', type=float, default=None)
    parser.add_argument('--max_joint_vel', type=float, default=None)
    parser.add_argument('--max_joint_acc', type=float, default=None)
    parser.add_argument('--max_joint_jerk', type=float, default=None)
    
    # Point cloud parameters
    parser.add_argument('--generate_pcd', action='store_true')
    parser.add_argument('--accumulate_pcd', action='store_true')
    parser.add_argument('--pcd_interval', type=int, default=10)
    parser.add_argument('--cam_name', type=str, default="camera1")
    
    args = parser.parse_args()
    
    run_cem_planner(
        num_dof=args.num_dof,
        num_batch=args.num_batch,
        num_steps=args.num_steps,
        maxiter_cem=args.maxiter_cem,
        maxiter_projection=args.maxiter_projection,
        w_pos=args.w_pos,
        w_rot=args.w_rot,
        w_col=args.w_col,
        num_elite=args.num_elite,
        timestep=args.timestep,
        initial_qpos=args.initial_qpos,
        target_names=args.targets,
        show_viewer=not args.no_viewer,
        cam_distance=args.cam_distance,
        show_contact_points=not args.no_contact_points,
        position_threshold=args.position_threshold,
        rotation_threshold=args.rotation_threshold,
        save_data=args.save_data,
        data_dir=args.data_dir,
        stop_at_final_target=not args.continue_after_final,
        inference=args.inference,
        rnn=args.rnn,
        max_joint_pos=args.max_joint_pos,
        max_joint_vel=args.max_joint_vel,
        max_joint_acc=args.max_joint_acc,
        max_joint_jerk=args.max_joint_jerk,
        generate_pcd=args.generate_pcd,
        accumulate_pcd=args.accumulate_pcd,
        pcd_interval=args.pcd_interval,
        cam_name=args.cam_name
    )