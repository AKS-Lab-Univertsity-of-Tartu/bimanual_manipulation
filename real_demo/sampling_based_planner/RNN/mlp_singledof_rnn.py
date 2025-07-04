import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#GRU layer class
class CustomGRULayer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Custom GRU layer with output transformation for single sequence element
		
		Args:
			input_size (int): Size of input features
			hidden_size (int): Size of hidden state in GRU
			output_size (int): Size of the output after transformation
		"""
		super(CustomGRULayer, self).__init__()
		
		# GRU cell for processing input
		self.gru_cell = nn.GRUCell(input_size, hidden_size)
		
		# Transformation layer to generate output from hidden state
		self.output_transform = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size)
		)
		
		self.hidden_size = hidden_size
		
	def forward(self, x, h_t):
		"""
		Forward pass through the GRU layer
		
		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_size]
			context_vector (torch.Tensor): Context vector to initialize hidden state
										 Shape: [batch_size, hidden_size]
		
		Returns:
			tuple: (output, hidden_state)
				- output: tensor of shape [batch_size, output_size]
				- hidden_state: tensor of shape [batch_size, hidden_size]
		"""
		# Initialize hidden state with context vector
		# h_t = context_vector
		
		# Update hidden state with GRU cell
		h_t = self.gru_cell(x, h_t)
		
		# Transform hidden state to generate output
		g_t = self.output_transform(h_t)
		
		return g_t, h_t


class GRU_Hidden_State(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(GRU_Hidden_State, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out

#LSTM layer class
class CustomLSTMLayer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Custom LSTM layer with output transformation for single sequence element
		
		Args:
			input_size (int): Size of input features
			hidden_size (int): Size of hidden state in LSTM
			output_size (int): Size of the output after transformation
		"""
		super(CustomLSTMLayer, self).__init__()
		
        #In LSTM, hidden state = long term memory, cell state = short term memory
		# LSTM cell for processing input
		self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
		
		# Transformation layer to generate output from hidden state
		self.output_transform = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size)
		)
		
		self.hidden_size = hidden_size
		
	def forward(self, x, h_t, c_t):
		"""
		Forward pass through the LSTM layer
		
		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_size]
			h_t (torch.Tensor): Hidden state tensor of shape [batch_size, hidden_size]
			c_t (torch.Tensor): Cell state tensor of shape [batch_size, hidden_size]
		
		Returns:
			tuple: (output, hidden_state, cell_state)
				- output: tensor of shape [batch_size, output_size]
				- hidden_state: tensor of shape [batch_size, hidden_size]
				- cell_state: tensor of shape [batch_size, hidden_size]
		"""
		# Update hidden state and cell state with LSTM cell
		h_t, c_t = self.lstm_cell(x, (h_t, c_t))
		
		# Transform hidden state to generate output
		g_t = self.output_transform(h_t)
		
		return g_t, h_t, c_t


class LSTM_Hidden_State(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(LSTM_Hidden_State, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim * 2),  # Output both h_0 and c_0
		)
		
		self.hidden_dim = out_dim
	
	def forward(self, x):
		out = self.mlp(x)
		# Split output into hidden state and cell state
		h_0 = out[:, :self.hidden_dim]
		c_0 = out[:, self.hidden_dim:]
		return h_0, c_0


#MLP class
class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability

            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability
            
            nn.Linear(hidden_dim, 256),
            #nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability
            
            nn.Linear(256, out_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)


class MLPProjectionFilter(nn.Module):
    
    def __init__(self, mlp, rnn_context, rnn_init, num_batch, 
                 num_dof, num_steps, timestep, 
                 v_max, a_max, j_max, 
                 p_max, maxiter_projection, rnn):
        super(MLPProjectionFilter, self).__init__()
        
        # MLP Model
        self.mlp = mlp

        if rnn == 'GRU':
            self.gru_context = rnn_context
            self.gru_init = rnn_init
        elif rnn == 'LSTM':
            self.lstm_context = rnn_context
            self.lstm_init = rnn_init
        
        
        # Problem dimensions
        self.num_dof = num_dof
        self.num_steps = num_steps
        self.nvar_single = num_steps  # For velocity trajectory
        self.nvar = num_dof * num_steps
        self.num_batch = num_batch
        self.t = timestep
        
        # Constraint parameters (matching JAX version)
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.p_max = p_max
        

        # Projection parameters
        self.A_projection = torch.eye(self.nvar, device=device)
        self.rho_ineq = 1.0
        self.rho_projection = 1.0

        # Solver parameters
        self.maxiter_projection = maxiter_projection
        
        # Setup finite difference matrices
        self.setup_finite_difference_matrices()
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
            
        
        # Loss function
        self.rcl_loss = nn.MSELoss()

    def setup_finite_difference_matrices(self):
        """Create finite difference matrices matching JAX version"""
        # Create identity matrix for velocity (like P_vel)
        self.P_vel = torch.eye(self.num_steps, device=device)
        
        # Create acceleration matrix using diff
        vel_diff = torch.diff(self.P_vel, dim=0)
        self.P_acc = vel_diff / self.t
        
        # Create jerk matrix using diff of acceleration
        acc_diff = torch.diff(self.P_acc, dim=0)
        self.P_jerk = acc_diff / self.t
        
        # Create position matrix using cumsum
        self.P_pos = torch.cumsum(self.P_vel, dim=0) * self.t
        
        # Number of constraints
        self.num_acc = self.num_steps - 1
        self.num_jerk = self.num_acc - 1
        self.num_pos = self.num_steps

    def setup_optimization_matrices(self):
        """Setup matrices following JAX approach"""
        
        # For multi-DOF, we need to expand the single-DOF matrices
        # Velocity constraints: [P_vel; -P_vel] for each DOF
        A_vel_single = torch.vstack((self.P_vel, -self.P_vel))
        
        # Acceleration constraints: [P_acc; -P_acc] for each DOF
        A_acc_single = torch.vstack((self.P_acc, -self.P_acc))
        
        # Jerk constraints: [P_jerk; -P_jerk] for each DOF
        A_jerk_single = torch.vstack((self.P_jerk, -self.P_jerk))

        # Position constraints: [P_pos; -P_pos] for each DOF
        A_pos_single = torch.vstack((self.P_pos, -self.P_pos))
        
        # Expand for multiple DOFs using Kronecker product
        self.A_vel = torch.kron(torch.eye(self.num_dof, device=device), A_vel_single)
        self.A_acc = torch.kron(torch.eye(self.num_dof, device=device), A_acc_single)
        self.A_jerk = torch.kron(torch.eye(self.num_dof, device=device), A_jerk_single)
        self.A_pos = torch.kron(torch.eye(self.num_dof, device=device), A_pos_single)
        
        # Combined control matrix (like A_control in JAX)
        self.A_control = torch.vstack((self.A_vel, self.A_acc, self.A_jerk, self.A_pos))
        
        # Equality constraint matrix (boundary conditions)
        # # Constrain first and last velocity for each DOF
        # boundary_matrix = torch.tensor([
        #     [1.0] + [0.0] * (self.num_steps - 1),    # first timestep
        #     [0.0] * (self.num_steps - 1) + [1.0]     # last timestep
        # ], device=device)
                
        boundary_matrix = torch.tensor([
            [1.0] + [0.0] * (self.num_steps - 1)    # first timestep
        ], device=device)
        
        self.A_eq = torch.kron(torch.eye(self.num_dof, device=device), boundary_matrix)
        
        # Constraint dimensions
        self.num_vel_constraints = 2 * self.num_steps * self.num_dof
        self.num_acc_constraints = 2 * self.num_acc * self.num_dof
        self.num_jerk_constraints = 2 * self.num_jerk * self.num_dof
        self.num_pos_constraints = 2 * self.num_pos * self.num_dof
        self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + 
                                    self.num_jerk_constraints + self.num_pos_constraints)
        
        # Compute inverse of Q matrix for KKT system
        self.Q_inv = self._get_Q_inv()

    def _get_Q_inv(self):
        """Compute inverse of Q matrix for the KKT system"""
        cost = (self.A_projection.T @ self.A_projection + 
                self.rho_ineq * self.A_control.T @ self.A_control)
        
        # KKT system matrix
        Q_top = torch.hstack((cost, self.A_eq.T))
        Q_bottom = torch.hstack((self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)))
        Q = torch.vstack((Q_top, Q_bottom))
        
        return torch.inverse(Q)

    def compute_boundary_vec(self, v_start, v_goal):
        """Compute boundary condition vector"""
        # Stack start and goal velocities for all DOFs
        # v_start_batch = self.v_start.to(device) #self.v_start.unsqueeze(0).repeat(self.num_batch, 1)
        # v_goal_batch = self.v_goal.to(device) #self.v_goal.unsqueeze(0).repeat(self.num_batch, 1)
        v_start_batch = v_start
        v_goal_batch = v_goal
        # b_eq = torch.hstack([v_start_batch, v_goal_batch])
        b_eq = v_start_batch
        return b_eq

    def compute_feasible_control(self, xi_samples, s, xi_projected, lamda, theta_init, v_start, v_goal):
        """Compute feasible control following JAX approach exactly"""
        b_vel = torch.hstack((
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device),
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device)
        ))
        
        b_acc = torch.hstack((
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device),
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device)
        ))
        
        b_jerk = torch.hstack((
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device),
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device)
        ))
        
        b_pos = torch.hstack((
            (- theta_init + self.p_max) * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device),
            (  theta_init + self.p_max) * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device)
        ))

        b_control = torch.hstack((b_vel, b_acc, b_jerk, b_pos))
        
        # Augmented bounds with slack variables
        b_control_aug = b_control - s
        
        # Boundary conditions
        b_eq = self.compute_boundary_vec(v_start=v_start,v_goal=v_goal)

        # print("b_control_aug", b_control_aug.shape)
        # print("self.A_control", self.A_control.shape)

        # Linear cost term (following JAX)
        lincost = (-lamda - 
                  torch.matmul(self.A_projection.T, xi_samples.T).T - 
                  self.rho_ineq * torch.matmul(self.A_control.T, b_control_aug.T).T)
        
        # Solve KKT system
        rhs = torch.hstack((-lincost, b_eq))
        sol = torch.matmul(self.Q_inv, rhs.T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following JAX)
        s = torch.maximum(
            torch.zeros((self.num_batch, self.num_total_constraints), device=device),
            -torch.matmul(self.A_control, xi_projected.T).T + b_control
        )
        
        # Compute residual (following JAX)
        res_vec = torch.matmul(self.A_control, xi_projected.T).T - b_control + s
        res_norm = torch.linalg.norm(res_vec, dim=1)
        
        # Update Lagrange multipliers (following JAX)
        lamda = lamda - self.rho_ineq * torch.matmul(self.A_control.T, res_vec.T).T
        
        return xi_projected, s, res_norm, lamda

    def compute_projection_gru(self, xi_samples, xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, theta_init, v_start, v_goal, h_0):
        """Project sampled trajectories following JAX approach"""
        
        # Initialize variables
        xi_projected_init = xi_projected_output_nn
        lamda_init = lamda_init_nn_output
        #lamda_init = torch.zeros((self.num_batch, self.nvar), device=device)
        
        # Initialize slack variables
        s_init = s_init_nn_output 
        
        # Initialize tracking variables
        xi_projected = xi_projected_init
        lamda = lamda_init
        s = s_init
        h = h_0
        
        primal_residuals = []
        fixed_point_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter_projection):
            xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda = self.compute_feasible_control(
                xi_samples, s, xi_projected, lamda, theta_init, v_start, v_goal)
            
            #Perform GRU accelaration after fixed-point iteration, i.e., projection step
            r_1 = torch.hstack((s_prev, lamda_prev))
            r_2 = torch.hstack((s, lamda))
            r = torch.hstack((r_1, r_2, r_2 -r_1))

            gru_output, h = self.gru_context(r,h)

            s_delta = gru_output[:, 0: self.num_total_constraints]
            lamda_delta = gru_output[:, self.num_total_constraints: self.num_total_constraints+self.nvar]

            lamda = lamda+lamda_delta 
            s = s+s_delta
            s = torch.maximum( torch.zeros(( self.num_batch, self.num_total_constraints), device = device), s)

            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (torch.linalg.norm(lamda_prev - lamda, dim=1) + 
                                  torch.linalg.norm(s_prev - s, dim=1))
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)

        
        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        
        return xi_projected, primal_residuals, fixed_point_residuals
    
    def compute_projection_lstm(self, xi_samples, xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, theta_init, v_start, v_goal, h_0, c_0):
        """Project sampled trajectories following JAX approach"""
        
        # Initialize variables
        xi_projected_init = xi_projected_output_nn
        lamda_init = lamda_init_nn_output
        #lamda_init = torch.zeros((self.num_batch, self.nvar), device=device)
        
        # Initialize slack variables
        s_init = s_init_nn_output 
        
        # Initialize tracking variables
        xi_projected = xi_projected_init
        lamda = lamda_init
        s = s_init
        h = h_0
        c = c_0

        
        primal_residuals = []
        fixed_point_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter_projection):
            xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda = self.compute_feasible_control(
                xi_samples, s, xi_projected, lamda, theta_init, v_start, v_goal)
            
            #Perform GRU accelaration after fixed-point iteration, i.e., projection step
            r_1 = torch.hstack((s_prev, lamda_prev))
            r_2 = torch.hstack((s, lamda))
            r = torch.hstack((r_1, r_2, r_2 -r_1))

            lstm_output, h, c = self.lstm_context(r, h, c)

            s_delta = lstm_output[:, 0: self.num_total_constraints]
            lamda_delta = lstm_output[:, self.num_total_constraints: self.num_total_constraints+self.nvar]

            lamda = lamda+lamda_delta 
            s = s+s_delta
            s = torch.maximum( torch.zeros(( self.num_batch, self.num_total_constraints), device = device), s)

            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (torch.linalg.norm(lamda_prev - lamda, dim=1) + 
                                  torch.linalg.norm(s_prev - s, dim=1))
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)

        
        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        
        return xi_projected, primal_residuals, fixed_point_residuals
    
    def decoder_function(self, inp_norm, input_nn, theta_init, v_start, v_goal, rnn):
        """Decoder function to compute initials from normalized input"""
        # Get neural network output
        neural_output_batch = self.mlp(inp_norm)
        
        # For simplicity, use neural output as initial guess
        # In practice, you might want to structure this differently
        xi_projected_output_nn = neural_output_batch[:, :self.nvar]
        lamda_init_nn_output = neural_output_batch[:, self.nvar: 2*self.nvar]
        s_init_nn_output = neural_output_batch[:, 2*self.nvar: 2*self.nvar + self.num_total_constraints]

        s_init_nn_output = torch.maximum( torch.zeros(( self.num_batch, self.num_total_constraints ), device = device), s_init_nn_output)
        
        xi_samples_input_nn = input_nn[:, 0:self.nvar].to(device)

        if rnn == "GRU":
            h_0 = self.gru_init(inp_norm)
            xi_projected, primal_residuals, fixed_point_residuals = self.compute_projection_gru(
            xi_samples_input_nn, xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, theta_init, v_start, v_goal, h_0)
        
        elif rnn == "LSTM":
            h_0, c_0 = self.lstm_init(inp_norm)
            xi_projected, primal_residuals, fixed_point_residuals = self.compute_projection_lstm(
            xi_samples_input_nn, xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, theta_init, v_start, v_goal, h_0, c_0)


        # Run projection
        

        
        # Compute average residuals
        # avg_res_primal = torch.mean(primal_residuals, dim=0)
        # avg_res_fixed_point = torch.mean(fixed_point_residuals, dim=0)

        avg_res_primal = torch.sum(primal_residuals, dim = 0)/self.maxiter_projection
        avg_res_fixed_point = torch.sum(fixed_point_residuals, dim = 0)/self.maxiter_projection
        
        return xi_projected, avg_res_fixed_point, avg_res_primal, primal_residuals, fixed_point_residuals

    def mlp_loss(self, avg_res_primal, avg_res_fixed_point, xi_samples_input_nn, xi_projected_output_nn):
        """Compute loss for optimization"""
        # Component losses
        primal_loss = 0.5 * torch.mean(avg_res_primal)
        fixed_point_loss = 0.5 * torch.mean(avg_res_fixed_point)
        projection_loss = 0.5* self.rcl_loss(xi_projected_output_nn, xi_samples_input_nn)

        # Total loss
        loss = primal_loss + fixed_point_loss + 1.0*projection_loss

        return primal_loss, fixed_point_loss, projection_loss, loss

    def forward(self, input_nn,  theta_init, v_start, v_goal, rnn):
        """Forward pass through the model"""
        # Normalize input with mean and standard deviation
        # inp_mean = input_nn.mean()
        # inp_std = input_nn.std()
        # inp_norm = (input_nn - inp_mean) / inp_std

        #Normalize input with mdian and quartile: Robust scaling
        inp_median_ = torch.median(input_nn, dim=0).values
        inp_q1 = torch.quantile(input_nn, 0.25, axis=0)
        inp_q3 = torch.quantile(input_nn, 0.75, axis=0)
        inp_iqr_ = inp_q3 - inp_q1
        # Handle constant features
        inp_iqr_ = torch.where(inp_iqr_ == 0, torch.tensor(1.0), inp_iqr_)
        inp_norm = (input_nn - inp_median_) / inp_iqr_


        # Decode input to get control
        xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = self.decoder_function(
            inp_norm, input_nn, theta_init, v_start, v_goal, rnn)
            
        return xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history