import os
from ament_index_python.packages import get_package_share_directory


xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mj_planner import bernstein_coeff_order10_arbitinterval

from functools import partial
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx 
import jax
import time


class cem_planner():

	def __init__(self, num_dof=6, num_batch=100, num_steps=200, timestep=0.02, maxiter_cem=20, num_elite=0.1, w_pos=2, w_rot=0.03, w_col=0.1):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps
		self.num_elite = num_elite
		self.cost_weights = {
			'w_pos': w_pos,
			'w_rot': w_rot,
			'w_col': w_col,
		}

		self.t_fin = self.num*self.t
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.A_projection = jnp.identity(self.nvar)
		self.rho_ineq = 1.0
		self.rho_projection = 1.0
		
		A_v_ineq, A_v = self.get_A_v()
		self.A_v_ineq = jnp.asarray(A_v_ineq) 
		self.A_v = jnp.asarray(A_v)

		A_a_ineq, A_a = self.get_A_a()
		self.A_a_ineq = jnp.asarray(A_a_ineq) 
		self.A_a = jnp.asarray(A_a)
  
		A_p_ineq, A_p = self.get_A_p()
		self.A_p_ineq = jnp.asarray(A_p_ineq) 
		self.A_p = jnp.asarray(A_p)
  
		A_eq = self.get_A_eq()
		self.A_eq = jnp.asarray(A_eq)
  
		Q_inv = self.get_Q_inv(A_eq)
		self.Q_inv = jnp.asarray(Q_inv)
  
		A_theta, A_thetadot, A_thetaddot = self.get_A_traj()
		self.A_theta = jnp.asarray(A_theta)
		self.A_thetadot = jnp.asarray(A_thetadot)
		self.A_thetaddot = jnp.asarray(A_thetaddot)
		
		self.compute_boundary_vec_batch = (jax.vmap(self.compute_boundary_vec_single, in_axes = (0)  ))

		self.key= jax.random.PRNGKey(0)
		self.maxiter_projection = 10
		self.maxiter_cem = maxiter_cem

		self.v_max = 0.8
		self.a_max = 1.8
		self.p_max = 180*np.pi/180
  
		self.l_1 = 1.0
		self.l_2 = 1.0
		self.l_3 = 1.0
		self.ellite_num = int(self.num_elite*self.num_batch)

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 10
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		# self.model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
		self.model_path = os.path.join(
            get_package_share_directory('real_demo'),
            'ur5e_hande_mjx',
            'scene.xml'
        )
		self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)

		self.geom_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)])
		# self.mask = jnp.sum(jnp.isin(self.mjx_data.contact.geom, self.geom_ids), axis=1)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids), axis=1)
		# self.mask = jnp.where(self.mask==2, 0, self.mask)
		# self.mask = self.mask.astype(bool)

		self.hande_id = self.model.body(name="hande").id
		self.tcp_id = self.model.site(name="tcp").id
		self.obstacle_qpos_idx = self.model.body_mocapid[self.model.body(name='obstacle').id]

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0, None, None, None, None))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0))

		self.print_info()


	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num}',
			f'\n Time per trajectory: {self.t_fin}',
		)

		
	def quaternion_multiply(self, q1, q2):
		"""Multiply two quaternions q1 * q2"""
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
		x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
		y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
		z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
		
		return jnp.array([w, x, y, z])
		
  
	def get_A_traj(self):
		A_theta = np.kron(np.identity(self.num_dof), self.P )
		A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )
		return A_theta, A_thetadot, A_thetaddot	
	
	def get_A_p(self):
		A_p = np.vstack(( self.P, -self.P     ))
		A_p_ineq = np.kron(np.identity(self.num_dof), A_p )
		return A_p_ineq, A_p
	
	def get_A_v(self):
		A_v = np.vstack(( self.Pdot, -self.Pdot     ))
		A_v_ineq = np.kron(np.identity(self.num_dof), A_v )
		return A_v_ineq, A_v

	def get_A_a(self):
		A_a = np.vstack(( self.Pddot, -self.Pddot  ))
		A_a_ineq = np.kron(np.identity(self.num_dof), A_a )
		return A_a_ineq, A_a
	
	def get_A_eq(self):
		return np.kron(np.identity(self.num_dof), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], self.Pddot[-1]    )))
	
	def get_Q_inv(self, A_eq):
		Q_inv = np.linalg.inv(np.vstack((np.hstack(( np.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_v_ineq.T, self.A_v_ineq)+self.rho_ineq*jnp.dot(self.A_a_ineq.T, self.A_a_ineq)+self.rho_ineq*jnp.dot(self.A_p_ineq.T, self.A_p_ineq), A_eq.T)  ), 
									 np.hstack((A_eq, np.zeros((np.shape(A_eq)[0], np.shape(A_eq)[0])))))))	
		return Q_inv

	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):
		b_eq_term = state_term.reshape(5, self.num_dof).T
		b_eq_term = b_eq_term.reshape(self.num_dof*5)
		return b_eq_term

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection(self, lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples):
  
		v_max_temp = jnp.hstack(( self.v_max*jnp.ones((self.num_batch, self.num  )),  self.v_max*jnp.ones((self.num_batch, self.num  ))       ))
		v_max_vec = jnp.tile(v_max_temp, (1, self.num_dof)  )

		a_max_temp = jnp.hstack(( self.a_max*jnp.ones((self.num_batch, self.num  )),  self.a_max*jnp.ones((self.num_batch, self.num  ))       ))
		a_max_vec = jnp.tile(a_max_temp, (1, self.num_dof)  )
		
		p_max_temp = jnp.hstack(( self.p_max*jnp.ones((self.num_batch, self.num  )),  self.p_max*jnp.ones((self.num_batch, self.num  ))       ))
		p_max_vec = jnp.tile(p_max_temp, (1, self.num_dof)  )
  
		b_v = v_max_vec 
		b_a = a_max_vec 
		b_p = p_max_vec
		
		b_v_aug = b_v-s_v
		b_a_aug = b_a-s_a 
		b_p_aug = b_p-s_p
  
		lincost = -lamda_v-lamda_a-lamda_p-self.rho_projection*jnp.dot(self.A_projection.T, xi_samples.T).T-self.rho_ineq*jnp.dot(self.A_v_ineq.T, b_v_aug.T).T-self.rho_ineq*jnp.dot(self.A_a_ineq.T, b_a_aug.T).T-self.rho_ineq*jnp.dot(self.A_p_ineq.T, b_p_aug.T).T
		sol = jnp.dot(self.Q_inv, jnp.hstack(( -lincost, b_eq_term )).T).T
  
		primal_sol = sol[:, 0:self.nvar]
		s_v = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_v_ineq, primal_sol.T).T+b_v  )
		res_v = jnp.dot(self.A_v_ineq, primal_sol.T).T-b_v+s_v 

		s_a = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_a_ineq, primal_sol.T).T+b_v  )
		res_a = jnp.dot(self.A_a_ineq, primal_sol.T).T-b_a+s_a 

		s_p = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_p_ineq, primal_sol.T).T+b_p  )
		res_p = jnp.dot(self.A_p_ineq, primal_sol.T).T-b_p+s_p 
	
		lamda_v = lamda_v-self.rho_ineq*jnp.dot(self.A_v_ineq.T, res_v.T).T
		lamda_a = lamda_a-self.rho_ineq*jnp.dot(self.A_a_ineq.T, res_a.T).T
		lamda_p = lamda_p-self.rho_ineq*jnp.dot(self.A_p_ineq.T, res_p.T).T
  
		res_v_vec = jnp.linalg.norm(res_v, axis = 1)
		res_a_vec = jnp.linalg.norm(res_a, axis = 1)
		res_p_vec = jnp.linalg.norm(res_p, axis = 1)
		
		res_projection = res_v_vec+res_a_vec+res_p_vec
		
		return primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_filter(self, xi_samples, state_term):

		b_eq_term = self.compute_boundary_vec_batch(state_term)
		s_v = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_a = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_p = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		lamda_v = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_a = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_p = jnp.zeros(( self.num_batch, self.nvar  ))
		
		for i in range(0, self.maxiter_projection):
			primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection  = self.compute_projection(lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples)
	 
		return primal_sol

	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, thetadot_single):

		qvel = mjx_data.qvel.at[:self.num_dof].set(thetadot_single)
		mjx_data = mjx_data.replace(qvel=qvel)
		mjx_data = self.jit_step(self.mjx_model, mjx_data)

		theta = mjx_data.qpos[:self.num_dof]
		eef_rot = mjx_data.xquat[self.hande_id]	
		eef_pos = mjx_data.site_xpos[self.tcp_id]
		collision = mjx_data.contact.dist[self.mask]

		return mjx_data, (theta, eef_pos, eef_rot, collision)

	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single(self, thetadot, init_pos, init_vel, obstacle_pos, obstacle_rot):
		mjx_data = self.mjx_data
		qvel = mjx_data.qvel.at[:self.num_dof].set(init_vel)
		qpos = mjx_data.qpos.at[:self.num_dof].set(init_pos)
		mocap_pos = mjx_data.mocap_pos.at[self.obstacle_qpos_idx].set(obstacle_pos)
		mocap_quat = mjx_data.mocap_quat.at[self.obstacle_qpos_idx].set(obstacle_rot)
		mjx_data = mjx_data.replace(qvel=qvel, qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)
		thetadot_single = thetadot.reshape(self.num_dof, self.num)
		_, out = jax.lax.scan(self.mjx_step, mjx_data, thetadot_single.T, length=self.num)
		theta, eef_pos, eef_rot, collision = out
		return theta.T.flatten(), eef_pos, eef_rot, collision
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, thetadot, eef_pos, eef_rot, collision, target_pos, target_rot):
		cost_g_ = jnp.linalg.norm(eef_pos - target_pos, axis=1)
		cost_g = cost_g_[-1] + jnp.sum(cost_g_[:-1])*1

		dot_product = jnp.abs(jnp.dot(eef_rot/jnp.linalg.norm(eef_rot, axis=1).reshape(1, self.num).T, target_rot/jnp.linalg.norm(target_rot)))
		dot_product = jnp.clip(dot_product, -1.0, 1.0)
		cost_r_ = 2 * jnp.arccos(dot_product)
		cost_r = cost_r_[-1] + jnp.sum(cost_r_[:-1])*1

		y = 0.005
		collision = collision.T
		g = -collision[:, 1:]+collision[:, :-1]-y*collision[:, :-1]
		cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], g.shape[1], 1), axis=-1, initial=0)) + jnp.sum(jnp.where(collision<0, True, False))

		cost = self.cost_weights['w_pos']*cost_g + self.cost_weights['w_rot']*cost_r + self.cost_weights['w_col']*cost_c
		return cost, cost_g_, cost_c
	
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
	def cem_iter(self, carry, _):
		init_pos, init_vel, target_pos, target_rot, obstacle_pos, obstacle_rot, xi_mean, xi_cov, key, state_term = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov)
		xi_filtered = self.compute_projection_filter(xi_samples, state_term)

		thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T

		theta, eef_pos, eef_rot, collision = self.compute_rollout_batch(thetadot, init_pos, init_vel, obstacle_pos, obstacle_rot)
		cost_batch, cost_g_batch, cost_c_batch = self.compute_cost_batch(thetadot, eef_pos, eef_rot, collision, target_pos, target_rot)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)

		carry = (init_pos, init_vel, target_pos, target_rot, obstacle_pos, obstacle_rot, xi_mean, xi_cov, key, state_term)
		return carry, (cost_batch, cost_g_batch, cost_c_batch, thetadot, theta)

	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		init_pos=jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0]), 
		init_vel=jnp.zeros(6), 
		init_acc=jnp.zeros(6),
		target_pos=jnp.zeros(3),
		target_rot=jnp.zeros(4),
		obstacle_pos=jnp.zeros(3),
		obstacle_rot=jnp.zeros(4)
		):

		theta_init = jnp.tile(init_pos, (self.num_batch, 1))
		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))
		thetaddot_init = jnp.tile(init_acc, (self.num_batch, 1))
		thetadot_fin = jnp.zeros((self.num_batch, self.num_dof))
		thetaddot_fin = jnp.zeros((self.num_batch, self.num_dof))

		target_pos = jnp.tile(target_pos, (self.num_batch, 1))
		target_rot = jnp.tile(target_rot, (self.num_batch, 1))

		state_term = jnp.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))
		state_term = jnp.asarray(state_term)
		
		xi_cov = 50*jnp.identity(self.nvar)
  
		key, subkey = jax.random.split(self.key)

		carry = (init_pos, init_vel, target_pos, target_rot, obstacle_pos, obstacle_rot, xi_mean, xi_cov, key, state_term)
		scan_over = jnp.array([0]*self.maxiter_cem)
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, cost_g_batch, cost_c_batch, thetadot, theta = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_vels = thetadot[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_traj = theta[-1][idx_min].reshape((self.num_dof, self.num)).T
		best_cost_g = cost_g_batch[-1][idx_min]
		best_cost_c = cost_c_batch[-1][idx_min]
		xi_mean = carry[6]

		return cost, best_cost_g, best_cost_c, best_vels, best_traj, xi_mean
	
def main():
	num_dof = 6
	num_batch = 500

	start_time = time.time()
	opt_class = cem_planner(num_dof, num_batch, w_pos=3, num_elite=0.1, maxiter_cem=30)	

	start_time_comp_cem = time.time()
	cost, best_cost_g, best_vels, best_traj, xi_mean = opt_class.compute_cem()

	print(f"Total time: {round(time.time()-start_time, 2)}s")
	print(f"Compute CEM time: {round(time.time()-start_time_comp_cem, 2)}s")

	np.savetxt('data/output_costs.csv',cost, delimiter=",")
	np.savetxt('data/best_vels.csv',best_vels, delimiter=",")
	np.savetxt('data/best_traj.csv',best_traj, delimiter=",")
	np.savetxt('data/best_cost_g.csv',best_cost_g, delimiter=",")

	
	
if __name__ == "__main__":
	main()


  	
