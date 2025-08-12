import os
from ament_index_python.packages import get_package_share_directory


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


from functools import partial
import numpy as np

import mujoco
import mujoco.mjx as mjx 
import jax
import jax.numpy as jnp


class cem_planner():

	def __init__(self, model=None, num_dof=None, num_batch=None, num_steps=None, timestep=None, maxiter_cem=None, num_elite=None, 
			     maxiter_projection=None, max_joint_pos = None ,max_joint_vel = None, 
				 max_joint_acc = None, max_joint_jerk = None):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite

		self.t_fin = self.num*self.t
		self.init_joint_position = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)

		self.P = jnp.identity(self.num) # Velocity mapping 
		self.Pdot = jnp.diff(self.P, axis=0)/self.t # Accelaration mapping
		self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t # Jerk mapping
		self.Pint = jnp.cumsum(self.P, axis=0)*self.t # Position mapping
		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.Pint_jax = jnp.asarray(self.Pint)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.rho_ineq = 1.0
		self.rho_projection = 1.0

		self.A_projection_single_dof = jnp.identity(self.nvar_single)

		A_v_ineq_single_dof, A_v_single_dof = self.get_A_v_single_dof()
		self.A_v_ineq_single_dof = jnp.asarray(A_v_ineq_single_dof) 
		self.A_v_single_dof = jnp.asarray(A_v_single_dof)

		A_a_ineq_single_dof, A_a_single_dof = self.get_A_a_single_dof()
		self.A_a_ineq_single_dof = jnp.asarray(A_a_ineq_single_dof) 
		self.A_a_single_dof = jnp.asarray(A_a_single_dof)

		A_j_ineq_single_dof, A_j_single_dof = self.get_A_j_single_dof()
		self.A_j_ineq_single_dof = jnp.asarray(A_j_ineq_single_dof)
		self.A_j_single_dof = jnp.asarray(A_j_single_dof)
  
		A_p_ineq_single_dof, A_p_single_dof = self.get_A_p_single_dof()
		self.A_p_ineq_single_dof = jnp.asarray(A_p_ineq_single_dof) 
		self.A_p_single_dof = jnp.asarray(A_p_single_dof)

		# Combined control matrix (like A_control in )
		self.A_control_single_dof = jnp.vstack((
			self.A_v_ineq_single_dof,
			self.A_a_ineq_single_dof,
			self.A_j_ineq_single_dof,
			self.A_p_ineq_single_dof
		))

		A_eq_single_dof = self.get_A_eq_single_dof()
		self.A_eq_single_dof = jnp.asarray(A_eq_single_dof)

		A_theta, A_thetadot, A_thetaddot, A_thetadddot = self.get_A_traj()

		self.A_theta = np.asarray(A_theta)
		self.A_thetadot = np.asarray(A_thetadot)
		self.A_thetaddot = np.asarray(A_thetaddot)
		self.A_thetadddot = np.asarray(A_thetadddot)
		
		self.key= jax.random.PRNGKey(42)
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		self.v_max = max_joint_vel
		self.a_max = max_joint_acc
		self.j_max = max_joint_jerk
		self.p_max = max_joint_pos		
		    
    	# Calculating number of Inequality constraints
		self.num_vel = self.num
		self.num_acc = self.num - 1
		self.num_jerk = self.num - 2
		self.num_pos = self.num

		self.num_vel_constraints = 2 * self.num_vel * num_dof
		self.num_acc_constraints = 2 * self.num_acc * num_dof
		self.num_jerk_constraints = 2 * self.num_jerk * num_dof
		self.num_pos_constraints = 2 * self.num_pos * num_dof
		self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + self.num_jerk_constraints + self.num_pos_constraints)
		self.num_total_constraints_per_dof = 2*(self.num_vel + self.num_acc + self.num_jerk + self.num_pos)

		self.ellite_num = int(self.num_elite*self.num_batch)

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 10
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		self.model_path = os.path.join(get_package_share_directory('real_demo'), 'ur5e_hande_mjx', 'scene.xml')
		# self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.model = model
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)
		self.jit_forward = jax.jit(mjx.forward)

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

		self.joint_mask_pos = np.isin(np.array(joint_names_pos), robot_joints)
		self.joint_mask_vel = np.isin(np.array(joint_names_vel), robot_joints)

		self.geom_ids = []
		
		for i in range(self.model.ngeom):
			name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
			if name is not None and (
				name.startswith('robot') 
				# or
				# name.startswith('object') 
				# name.startswith('target')
			):  
				print(f"Found geom: id={i}, name='{name}'")
				self.geom_ids.append(i)

		self.geom_ids_all = np.array(self.geom_ids)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids_all), axis=1)

		self.hande_id_0 = self.model.body(name="hande_0").id
		self.tcp_id_0 = self.model.site(name="tcp_0").id

		self.hande_id_1 = self.model.body(name="hande_1").id
		self.tcp_id_1 = self.model.site(name="tcp_1").id

		self.object_0_site_0_id = self.model.site(name="object_0_site_0").id
		self.object_0_site_1_id = self.model.site(name="object_0_site_1").id

		self.obj_qpos_idx = self.mjx_model.body_dofadr[self.model.body(name="object_0").id]

		self.weld_id_0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_0")
		self.weld_id_1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_1")

		# self.tray_0_id = self.model.body(name="target_0").id
		# self.tray_1_id = self.model.body(name="target_1").id

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0, None, None, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
		self.compute_boundary_vec_batch_single_dof = (jax.vmap(self.compute_boundary_vec_single_dof, in_axes = (0)  )) # vmap parrallelization takes place over first axis
		self.compute_projection_batched_over_dof = jax.vmap(self.compute_projection_single_dof, in_axes=(0, 0, 0, 0, 0)) # vmap parrallelization takes place over first axis

		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Projection Iter: {self.maxiter_projection}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
			f'\n Number of variables: {self.nvar}',
			f'\n Number of Total constraints: {self.num_total_constraints}',
			f'\n Number of geomteric IDs for colllision: {len(self.geom_ids_all)}'
		    f'\n{self.mask.sum()} / {self.mask.shape[0]} contacts involve robot.'
		)

    
	def get_A_traj(self):

		# This is valid while dealing with knots anfd projecting into pos,vel,acc space with Bernstein Polynomials
		# A_theta = np.kron(np.identity(self.num_dof), self.P )
		# A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		# A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )
        
        # This is valid while not using knots and bernstein polynomials; directlly using velocity
		A_theta = np.kron(np.identity(self.num_dof), self.Pint )
		A_thetadot = np.kron(np.identity(self.num_dof), self.P )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetadddot = np.kron(np.identity(self.num_dof), self.Pddot )

		return A_theta, A_thetadot, A_thetaddot, A_thetadddot	


	def get_A_p_single_dof(self):
		A_p = np.vstack(( self.Pint, -self.Pint))
		A_p_ineq = np.kron(np.identity(1), A_p )
		return A_p_ineq, A_p
	
	def get_A_v_single_dof(self):
		A_v = np.vstack(( self.P, -self.P     ))
		A_v_ineq = np.kron(np.identity(1), A_v )
		return A_v_ineq, A_v

	def get_A_a_single_dof(self):
		A_a = np.vstack(( self.Pdot, -self.Pdot  ))
		A_a_ineq = np.kron(np.identity(1), A_a )
		return A_a_ineq, A_a
	
	def get_A_j_single_dof(self):
		A_j = np.vstack(( self.Pddot, -self.Pddot  ))
		A_j_ineq = np.kron(np.identity(1), A_j )
		return A_j_ineq, A_j
	
	def get_A_eq_single_dof(self):
		return np.kron(np.identity(1), self.P[0])

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single_dof(self, state_term):
		num_eq_constraint_per_dof = int(jnp.shape(state_term)[0])
		b_eq_term = state_term.reshape( num_eq_constraint_per_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint_per_dof)
		return b_eq_term
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control_single_dof(self, lamda_init_single_dof, s_init_single_dof, 
										 b_eq_term_single_dof, xi_samples_single_dof, 
										 init_pos_single_dof):
		b_vel = jnp.hstack((
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof))),
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof)))
		))

		b_acc = jnp.hstack((
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof))),
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof)))
		))

		b_jerk = jnp.hstack((
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof))),
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof)))
		))
        

		init_pos_single_dof_batch = jnp.tile(init_pos_single_dof, (self.num_batch, 1))  # (num_batch, 1)
        
		# Calculate bounds for each joint and each batch
    	# Upper bounds: p_max - init_pos, Lower bounds: p_max + init_pos (assuming symmetric limits)
		b_pos_upper = (self.p_max - init_pos_single_dof_batch)  # shape (num_batch, 1)
		b_pos_lower = (self.p_max + init_pos_single_dof_batch)  # shape (num_batch, 1)
        
		
		# Expand to include time steps
		b_pos_upper_expanded = jnp.tile(b_pos_upper[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraints per dof/2)
		b_pos_lower_expanded = jnp.tile(b_pos_lower[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))  # (num_batch, 1, num_pos_constraintsper dof/2)
		
		# Stack upper and lower bounds
		b_pos_stacked = jnp.concatenate([b_pos_upper_expanded, b_pos_lower_expanded], axis=2)  # (num_batch, 1, num_pos_constraints per dof)
		
		# Reshape to final form: (num_batch, total_pos_constraints)
		b_pos = b_pos_stacked.reshape((self.num_batch, -1))  # shape: (num_batch, self.num_pos_constraints per dof)
        
		b_control_single_dof = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))

		# Augmented bounds with slack variables
		b_control_aug_single_dof = b_control_single_dof - s_init_single_dof

		# Cost matrix
		cost = (
			jnp.dot(self.A_projection_single_dof.T, self.A_projection_single_dof) +
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, self.A_control_single_dof)
		)

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq_single_dof.T)),
			jnp.hstack((self.A_eq_single_dof, jnp.zeros((self.A_eq_single_dof.shape[0], self.A_eq_single_dof.shape[0]))))
		))

		# Linear cost term
		lincost = (
			-lamda_init_single_dof -
			jnp.dot(self.A_projection_single_dof.T, xi_samples_single_dof.T).T -
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, b_control_aug_single_dof.T).T
		)

		# Solve KKT system
		sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_term_single_dof)).T).T

		# Extract primal solution
		xi_projected = sol[:, :self.nvar_single]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints_per_dof)),
			-jnp.dot(self.A_control_single_dof, xi_projected.T).T + b_control_single_dof
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control_single_dof, xi_projected.T).T - b_control_single_dof + s
		res_norm = jnp.linalg.norm(res_vec, axis=1)

		# Update Lagrange multipliers
		lamda = lamda_init_single_dof - self.rho_ineq * jnp.dot(self.A_control_single_dof.T, res_vec.T).T

		return xi_projected, s, res_norm, lamda
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_single_dof(self, 
								       xi_samples_single_dof, 
								       state_term_single_dof, 
									   lamda_init_single_dof, 
									   s_init_single_dof, 
									   init_pos_single_dof):
		
		# state_term_single_dof: (B, K) → flatten across batch
		b_eq_term = self.compute_boundary_vec_batch_single_dof(state_term_single_dof)  # should become (B, K), flattened

		xi_projected_init_single_dof = xi_samples_single_dof

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control_single_dof(lamda, 
																		s, b_eq_term, xi_samples_single_dof, 
																		init_pos_single_dof)
			
			primal_residual = res_projection
			fixed_point_residual = (
				jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				jnp.linalg.norm(s_prev - s, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init_single_dof, lamda_init_single_dof, s_init_single_dof)


		carry_final, res_tot = jax.lax.scan(
			lax_custom_projection,
			carry_init,
			jnp.arange(self.maxiter_projection)
		)

		primal_sol, lamda, s = carry_final
		primal_residuals, fixed_point_residuals = res_tot

		primal_residuals = jnp.stack(primal_residuals)
		fixed_point_residuals = jnp.stack(fixed_point_residuals)

		return primal_sol, primal_residuals, fixed_point_residuals
	
	@partial(jax.jit, static_argnums=(0,))
	def rotmat_to_quat(self, mat):
		"""
		Convert a 3x3 rotation matrix to a quaternion (w, x, y, z) using JAX.
		Assumes the matrix is a valid rotation matrix.
		"""
		m = mat.reshape((3, 3))
		tr = m[0, 0] + m[1, 1] + m[2, 2]

		def case_tr_pos(_):
			S = jnp.sqrt(tr + 1.0) * 2  # S=4*w
			w = 0.25 * S
			x = (m[2, 1] - m[1, 2]) / S
			y = (m[0, 2] - m[2, 0]) / S
			z = (m[1, 0] - m[0, 1]) / S
			return jnp.array([w, x, y, z])

		def case_m00_max(_):
			S = jnp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S=4*x
			w = (m[2, 1] - m[1, 2]) / S
			x = 0.25 * S
			y = (m[0, 1] + m[1, 0]) / S
			z = (m[0, 2] + m[2, 0]) / S
			return jnp.array([w, x, y, z])

		def case_m11_max(_):
			S = jnp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S=4*y
			w = (m[0, 2] - m[2, 0]) / S
			x = (m[0, 1] + m[1, 0]) / S
			y = 0.25 * S
			z = (m[1, 2] + m[2, 1]) / S
			return jnp.array([w, x, y, z])

		def case_m22_max(_):
			S = jnp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S=4*z
			w = (m[1, 0] - m[0, 1]) / S
			x = (m[0, 2] + m[2, 0]) / S
			y = (m[1, 2] + m[2, 1]) / S
			z = 0.25 * S
			return jnp.array([w, x, y, z])

		quat = jnp.where(tr > 0, case_tr_pos(0),
				jnp.where((m[0, 0] > m[1, 1]) & (m[0, 0] > m[2, 2]), case_m00_max(0),
				jnp.where(m[1, 1] > m[2, 2], case_m11_max(0), case_m22_max(0))
			))

		return quat
	
	@partial(jax.jit, static_argnums=(0,))  
	def angle_between_lines(self, p1, p2, p3, p4):
		"""
		Calculates the signed angle between two lines using JAX.
		This version is JIT-compatible and uses jax.lax.cond for branching.
		"""
		# Create vectors from the points (p1->p2 and p3->p4)
		v1 = jnp.array([p2[0] - p1[0], p2[1] - p1[1]])
		v2 = jnp.array([p4[0] - p3[0], p4[1] - p3[1]])

		# --- Define the functions for our conditional logic ---

		def calculate_angle(operands):
			"""The main logic path."""
			v1_op, v2_op = operands
			angle1 = jnp.arctan2(v1_op[1], v1_op[0])
			angle2 = jnp.arctan2(v2_op[1], v2_op[0])
			angle_rad = angle2 - angle1
			return jnp.degrees(angle_rad)

		def return_zero(operands):
			"""The exception path."""
			# Must return the same shape and dtype as the other branch
			return 0.0

		def check_parallel_and_calculate(operands):
			"""Nested check for the dot product."""
			v1_op, v2_op = operands
			
			# Normalize vectors for the dot product check
			norm_v1 = jnp.linalg.norm(v1_op)
			norm_v2 = jnp.linalg.norm(v2_op)
			u1 = v1_op / norm_v1
			u2 = v2_op / norm_v2
			
			dot_product = jnp.dot(u1, u2)
			
			# Second condition: Are the vectors parallel?
			is_parallel = jnp.abs(dot_product - 1.0) < 0.00001
			
			return jax.lax.cond(
				is_parallel,
				return_zero,          # If parallel, return 0
				calculate_angle,      # If not parallel, calculate the angle
				operands
			)

		# --- Execute the conditional logic ---
		norm_v1 = jnp.linalg.norm(v1)
		norm_v2 = jnp.linalg.norm(v2)
		epsilon = 0.1

		# First condition: Are the vectors long enough?
		is_too_short = (norm_v1 < epsilon) | (norm_v2 < epsilon)

		return jax.lax.cond(
			is_too_short,
			return_zero,                      # If too short, return 0
			check_parallel_and_calculate,     # If long enough, proceed to the next check
			(v1, v2)                          # The operands passed to the chosen function
		)
	
	@partial(jax.jit, static_argnums=(0,))
	def quaternion_distance(self, q1, q2):
		dot_product = jnp.abs(jnp.dot(q1, q2))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		return 2 * jnp.arccos(dot_product)

	@partial(jax.jit, static_argnums=(0,))
	def rotation_quaternion(self, angle_deg, axis):
		axis = axis / jnp.linalg.norm(axis)
		angle_rad = jnp.deg2rad(angle_deg)
		w = jnp.cos(angle_rad / 2)
		x, y, z = axis * jnp.sin(angle_rad / 2)
		return jnp.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])

	@partial(jax.jit, static_argnums=(0,))
	def quaternion_multiply(self, q1, q2):
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
		x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
		y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
		z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
		
		return jnp.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])
	
	@partial(jax.jit, static_argnums=(0,))
	def turn_tray(self, eef_pos_0_init, eef_pos_1_init, eef_pos_0, eef_pos_1, tray_rot_init):
		# xy plane z-axis rotation
		p1, p2 = (eef_pos_0_init[0], eef_pos_0_init[1]), (eef_pos_1_init[0], eef_pos_1_init[1])
		p3, p4 = (eef_pos_0[0], eef_pos_0[1]), (eef_pos_1[0], eef_pos_1[1])
		z_rot = self.angle_between_lines(p1, p2, p3, p4)
		tray_rot = self.quaternion_multiply(tray_rot_init, self.rotation_quaternion(z_rot, jnp.array([0, 0, 1])))

		# # # xz plane y-axis rotation
		# p1, p2 = (eef_pos_0_init[2], eef_pos_0_init[0]), (eef_pos_1_init[2], eef_pos_1_init[0])
		# p3, p4 = (eef_pos_0[2], eef_pos_0[0]), (eef_pos_1[2], eef_pos_1[0])
		# y_rot = self.angle_between_lines(p1, p2, p3, p4)
		# tray_rot = self.quaternion_multiply(tray_rot, self.rotation_quaternion(y_rot, jnp.array([0, 1, 0])))

		# # # yz plane x-axis rotation
		# p1, p2 = (eef_pos_0_init[1], eef_pos_0_init[2]), (eef_pos_1_init[1], eef_pos_1_init[2])
		# p3, p4 = (eef_pos_0[1], eef_pos_0[2]), (eef_pos_1[1], eef_pos_1[2])
		# x_rot = self.angle_between_lines(p1, p2, p3, p4)
		# tray_rot = self.quaternion_multiply(tray_rot, self.rotation_quaternion(x_rot, jnp.array([1, 0, 0])))

		return tray_rot




	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):
	
		qvel = mjx_data.qvel.at[self.joint_mask_vel].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		
		# Step the simulation
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		# Get joint positions and end-effector states
		theta = mjx_data.qpos[self.joint_mask_pos]
		
		# First arm end-effector 
		eef_pos_0 = mjx_data.site_xpos[self.tcp_id_0]
		eef_rot_0 = mjx_data.xquat[self.hande_id_0]   
		eef_0 = jnp.concatenate([eef_pos_0, eef_rot_0])
		
		# Second arm end-effector
		eef_pos_1 = mjx_data.site_xpos[self.tcp_id_1]
		eef_rot_1 = mjx_data.xquat[self.hande_id_1]    
		eef_1 = jnp.concatenate([eef_pos_1, eef_rot_1])
		
		# Collision detection
		collision = mjx_data.contact.dist[self.mask]

		# Set tray position and orientation
		obj_pos_0 = mjx_data.site_xpos[self.object_0_site_0_id]
		obj_pos_1 = mjx_data.site_xpos[self.object_0_site_1_id]

		obj_rot_0 = self.rotmat_to_quat(mjx_data.site_xmat[self.object_0_site_0_id])
		obj_rot_1 = self.rotmat_to_quat(mjx_data.site_xmat[self.object_0_site_1_id])

		obj_0 = jnp.concatenate([obj_pos_0, obj_rot_0])
		obj_1 = jnp.concatenate([obj_pos_1, obj_rot_1])

		obj = mjx_data.qpos[self.obj_qpos_idx : self.obj_qpos_idx+7]

		# Compute Jacobians using the current MJX API
		def get_site_pos0(qpos):
			# Create new data with updated qpos
			new_data = mjx_data.replace(qpos=qpos)
			# Forward kinematics
			new_data = mjx.forward(self.mjx_model, new_data)
			return new_data.site_xpos[self.tcp_id_0]
		
		def get_site_rot0(qpos):
			new_data = mjx_data.replace(qpos=qpos)
			new_data = mjx.forward(self.mjx_model, new_data)
			return new_data.xquat[self.hande_id_0]
		
		def get_site_pos1(qpos):
			new_data = mjx_data.replace(qpos=qpos)
			new_data = mjx.forward(self.mjx_model, new_data)
			return new_data.site_xpos[self.tcp_id_1]
		
		def get_site_rot1(qpos):
			new_data = mjx_data.replace(qpos=qpos)
			new_data = mjx.forward(self.mjx_model, new_data)
			return new_data.xquat[self.hande_id_1]

		# Compute Jacobians using JAX's automatic differentiation
		jacp0 = jax.jacfwd(get_site_pos0)(mjx_data.qpos)
		jacr0 = jax.jacfwd(get_site_rot0)(mjx_data.qpos)
		jacp1 = jax.jacfwd(get_site_pos1)(mjx_data.qpos)
		jacr1 = jax.jacfwd(get_site_rot1)(mjx_data.qpos)

		# Compute EEF velocities
		eef_vel_lin_0 = jacp0[:, self.joint_mask_pos] @ mjx_data.qvel[self.joint_mask_vel]
		eef_vel_ang_0 = jacr0[:, self.joint_mask_pos] @ mjx_data.qvel[self.joint_mask_vel]
		eef_vel_lin_1 = jacp1[:, self.joint_mask_pos] @ mjx_data.qvel[self.joint_mask_vel]
		eef_vel_ang_1 = jacr1[:, self.joint_mask_pos] @ mjx_data.qvel[self.joint_mask_vel]

		return mjx_data, (
			theta, 
			eef_0, eef_vel_lin_0, eef_vel_ang_0,
			eef_1, eef_vel_lin_1, eef_vel_ang_1, 
			obj, obj_0, obj_1, collision
		)


	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot, init_pos, init_vel, obj_init, grippers):

		mjx_data = self.mjx_data
		qvel = mjx_data.qvel.at[self.joint_mask_vel].set(init_vel)
		qpos = mjx_data.qpos.at[self.joint_mask_pos].set(init_pos)
		qpos = qpos.at[self.obj_qpos_idx : self.obj_qpos_idx + 7].set(obj_init)
		eq_active = mjx_data.eq_active.at[self.weld_id_0].set(grippers[0])
		eq_active = eq_active.at[self.weld_id_1].set(grippers[1])
		mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos, eq_active=eq_active)


		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		_, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
		theta, eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, eef_vel_lin_1, eef_vel_ang_1, obj, obj_0, obj_1, collision = out
		return theta.T.flatten(), eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, eef_vel_lin_1, eef_vel_ang_1, obj, obj_0, obj_1, collision
	

	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, theta, eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, eef_vel_lin_1, eef_vel_ang_1, obj, obj_0, obj_1, collision, target_0, cost_weights):
		# eef_0 = [x, y, z, w, x, y, z]

		''' Common cost for both tasks '''

		# Compute collision cost
		y = 0.15 # Higher y implies stricter condition on g to be positive
		collision = collision.T
		g = -collision[:, 1:]+(1 - y)*collision[:, :-1]
		cost_c = jnp.sum(jnp.maximum(g, 0)) + jnp.sum(collision < 0)

		# Keep arm at all times closer to the initial state
		cost_theta = jnp.linalg.norm(theta.reshape((self.num_dof, self.num)).T - self.init_joint_position)

		rel_pos = eef_0[:,:3] - eef_1[:,:3]        # Shape (Batch, 3)
		rel_vel = eef_vel_lin_0 - eef_vel_lin_1 # Shape (Batch, 3)

		dot_products = jnp.sum(rel_pos * rel_vel, axis=-1)  # Shape (Batch,)
		cost_eef_vel = jnp.linalg.norm(dot_products)

		# Move end effectors to pick positions
		cost_g_0 = jnp.sum(jnp.linalg.norm(eef_0[:, :3] - obj_0[:, :3], axis=1))
		# cost_g_0 = jnp.where(cost_g_0 > 0.5, 0, cost_g_0)
		cost_g_1 = jnp.sum(jnp.linalg.norm(eef_1[:, :3] - obj_1[:, :3], axis=1))
		# cost_g_1 = jnp.where(cost_g_1 > 0.5, 0, cost_g_1)
		# cost_g = jnp.min(jnp.array([cost_g_0, cost_g_1]))*0.8 + jnp.max(jnp.array([cost_g_0, cost_g_1]))*0.2
		cost_g = (cost_g_0+cost_g_1)/2

		# Move end effectors to pick orientation
		dot_product = jnp.abs(jnp.dot(eef_0[:, 3:]/jnp.linalg.norm(eef_0[:, 3:], axis=1).reshape(1, self.num).T, (obj_0[:, 3:]/jnp.linalg.norm(obj_0[:, 3:], axis=1).reshape(1, self.num).T).T))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_0 = 2 * jnp.arccos(dot_product)

		dot_product = jnp.abs(jnp.dot(eef_1[:, 3:]/jnp.linalg.norm(eef_1[:, 3:], axis=1).reshape(1, self.num).T, (obj_1[:, 3:]/jnp.linalg.norm(obj_1[:, 3:], axis=1).reshape(1, self.num).T).T))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_1 = 2 * jnp.arccos(dot_product)

		cost_r = (jnp.sum(cost_r_0) + jnp.sum(cost_r_1))/2


		# Move tray to target position
		cost_g_obj = jnp.linalg.norm(obj[:, :3] - target_0[:3])

		dot_product = jnp.abs(jnp.dot(obj[:, 3:]/jnp.linalg.norm(obj[:, 3:], axis=1).reshape(1, self.num).T, target_0[3:]/jnp.linalg.norm(target_0[3:])))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_obj = 2 * jnp.arccos(dot_product)
		cost_r_obj = jnp.sum(cost_r_obj)

		cost = (
			cost_weights['collision']*cost_c +
			cost_weights['theta']*cost_theta +
			# cost_weights['velocity']*cost_eef_vel +
			# cost_weights['z-axis']*cost_eef_pos +

			cost_weights['position']*cost_g +
			cost_weights['orientation']*cost_r+

			cost_weights['position_obj']*cost_g_obj+
			cost_weights['orientation_obj']*cost_r_obj 
		)	

		cost_list = jnp.array([
			cost_c, 
			cost_weights['position_obj']*cost_g_obj,
			cost_weights['position']*cost_g, 
			cost_weights['orientation_obj']*cost_r_obj

		])

		return cost, cost_list
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.0001*jnp.identity(self.nvar)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def cem_iter(self, carry,  scan_over):

		xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_vel, target_0, obj_init, cost_weights, grippers = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		xi_samples_reshaped = xi_samples.reshape(self.num_batch, self.num_dof, self.num)
		xi_samples_batched_over_dof = jnp.transpose(xi_samples_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		state_term_reshaped = state_term.reshape(self.num_batch, self.num_dof, 1)
		state_term_batched_over_dof = jnp.transpose(state_term_reshaped, (1, 0, 2)) #Shape: (DoF, B, 1)

		lamda_init_reshaped = lamda_init.reshape(self.num_batch, self.num_dof, self.num)
		lamda_init_batched_over_dof = jnp.transpose(lamda_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num)

		s_init_reshaped = s_init.reshape(self.num_batch, self.num_dof, self.num_total_constraints_per_dof )
		s_init_batched_over_dof = jnp.transpose(s_init_reshaped, (1, 0, 2)) # shape: (DoF, B, num_total_constraints_per_dof)


		
        # Pass all arguments as positional arguments; not keyword arguments
		xi_filtered, primal_residuals, fixed_point_residuals = self.compute_projection_batched_over_dof(
			                                                     xi_samples_batched_over_dof, 
														         state_term_batched_over_dof, 
																 lamda_init_batched_over_dof, 
																 s_init_batched_over_dof, 
																 init_pos)
		
		xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1) # shape: (B, num*num_dof)
		
		primal_residuals = jnp.linalg.norm(primal_residuals, axis = 0)
		fixed_point_residuals = jnp.linalg.norm(fixed_point_residuals, axis = 0)
				
		avg_res_primal = jnp.sum(primal_residuals, axis = 0)/self.maxiter_projection
    	
		avg_res_fixed_point = jnp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection

		thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T


		theta, eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, eef_vel_lin_1, eef_vel_ang_1, obj, obj_0, obj_1, collision = self.compute_rollout_batch(thetadot, init_pos, init_vel, obj_init, grippers)
		cost_batch, cost_list_batch = self.compute_cost_batch(theta, eef_0, eef_vel_lin_0, eef_vel_ang_0, eef_1, eef_vel_lin_1, eef_vel_ang_1, obj, obj_0, obj_1, collision, target_0, cost_weights)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)
		xi_samples_new, key = self.compute_xi_samples(key, xi_mean, xi_cov)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples_new, init_pos, init_vel, target_0, obj_init, cost_weights, grippers)

		return carry, (cost_batch, cost_list_batch, thetadot, theta, 
				 avg_res_primal, avg_res_fixed_point, primal_residuals, fixed_point_residuals)
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		xi_cov,
		init_pos, 
		init_vel, 
		init_acc,
		target_0,
		lamda_init,
		s_init,
		xi_samples,
		cost_weights,
		obj_init,
		grippers
		):


		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))

		state_term = thetadot_init	
		
		key, subkey = jax.random.split(self.key)

		carry = (xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples, init_pos, init_vel, target_0, obj_init, cost_weights, grippers)
		scan_over = jnp.array([0]*self.maxiter_cem)
		
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_list_batch, thetadot, theta, avg_res_primal, avg_res_fixed, primal_residuals, fixed_point_residuals = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_vels = thetadot[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_traj = theta[-1][idx_min].reshape((self.num_dof, self.num)).T

		best_cost_list = cost_list_batch[-1][idx_min]

		xi_mean = carry[0]
		xi_cov = carry[1]

	    
		return (
			cost,
			best_cost_list,
			best_vels,
			best_traj,
			xi_mean,
			xi_cov,
			thetadot,
			theta,
			avg_res_primal,
			avg_res_fixed,
			primal_residuals,
			fixed_point_residuals,
			idx_min,
		)