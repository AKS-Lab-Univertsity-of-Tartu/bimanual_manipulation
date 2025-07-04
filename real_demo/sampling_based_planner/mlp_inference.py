import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from tqdm import trange, tqdm

# Import your model components
from RNN.mlp_singledof_rnn import MLP, MLPProjectionFilter, CustomGRULayer, GRU_Hidden_State, CustomLSTMLayer, LSTM_Hidden_State


def sample_uniform_trajectories(key, var_min, var_max, dataset_size, nvar):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(low=var_min, high=var_max, size=(dataset_size, nvar))
    return xi_samples, rng


def rnn_inference(rnn_type="LSTM", model_weights_path=None, dataset_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Problem setup
    num_dof = 1
    num_steps = 50
    timestep = 0.05
    v_max = 1.0
    a_max = 2.0
    j_max = 5.0
    p_max = 180.0 * np.pi / 180.0
    nvar_single = num_steps
    nvar = num_dof * nvar_single
    theta_init_min = 0.0
    theta_init_max = 2 * np.pi
    maxiter_projection = 20

    # Constraints
    num_acc = num_steps - 1
    num_jerk = num_acc - 1
    num_pos = num_steps
    num_vel_constraints = 2 * num_steps * num_dof
    num_acc_constraints = 2 * num_acc * num_dof
    num_jerk_constraints = 2 * num_jerk * num_dof
    num_pos_constraints = 2 * num_pos * num_dof
    num_total_constraints = (
        num_vel_constraints + num_acc_constraints +
        num_jerk_constraints + num_pos_constraints
    )

    # Test inputs (scalar repeated)
    theta_init_scalar = 0.0 * np.pi
    v_start_scalar = 0.0
    v_goal_scalar = 0.0
    theta_init = np.tile(theta_init_scalar, (dataset_size, 1))
    v_start = np.tile(v_start_scalar, (dataset_size, 1))
    v_goal = np.tile(v_goal_scalar, (dataset_size, 1))

    # Random xi input
    xi_samples, _ = sample_uniform_trajectories(42, -v_max, v_max, dataset_size, nvar)
    inp = np.hstack((xi_samples, theta_init, v_start, v_goal))
    inp_mean, inp_std = inp.mean(), inp.std()

    # Choose RNN type
    if rnn_type == "GRU":
        print("Using GRU model")
        rnn_input_size = 3 * num_total_constraints + 3 * nvar
        rnn_hidden_size = 512
        rnn_output_size = num_total_constraints + nvar
        rnn_context = CustomGRULayer(rnn_input_size, rnn_hidden_size, rnn_output_size)
        rnn_init = GRU_Hidden_State(inp.shape[1], 512, rnn_hidden_size)
    else:
        print("Using LSTM model")
        rnn_input_size = 3 * num_total_constraints + 3 * nvar
        rnn_hidden_size = 512
        rnn_output_size = num_total_constraints + nvar
        rnn_context = CustomLSTMLayer(rnn_input_size, rnn_hidden_size, rnn_output_size)
        rnn_init = LSTM_Hidden_State(inp.shape[1], 512, rnn_hidden_size)

    # MLP and overall model
    mlp = MLP(inp.shape[1], 1024, 2 * nvar + num_total_constraints)
    model = MLPProjectionFilter(
        mlp=mlp,
        rnn_context=rnn_context,
        rnn_init=rnn_init,
        num_batch=dataset_size,
        num_dof=num_dof,
        num_steps=num_steps,
        timestep=timestep,
        v_max=v_max,
        a_max=a_max,
        j_max=j_max,
        p_max=p_max,
        maxiter_projection=maxiter_projection,
        rnn=rnn_type
    ).to(device)

    # Load weights
    if model_weights_path is None:
        model_weights_path = f'./training_weights/mlp_learned_single_dof_{rnn_type}.pth'
    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    model.eval()

    # Normalize input
    inp_test = torch.tensor(inp).float().to(device)
    inp_norm_test = (inp_test - inp_mean) / inp_std
    xi_samples_input_nn_test = inp_test
    theta_init_test = torch.tensor(theta_init).float().to(device)
    v_start_test = torch.tensor(v_start).float().to(device)
    v_goal_test = torch.tensor(v_goal).float().to(device)

    # Run model
    with torch.no_grad():
        xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = model.decoder_function(
            inp_norm_test,
            xi_samples_input_nn_test,
            theta_init_test,
            v_start_test,
            v_goal_test,
            rnn_type
        )

    return {
        "xi_projected": xi_projected,
        "avg_res_fixed_point": avg_res_fixed_point,
        "avg_res_primal": avg_res_primal,
        "res_primal_history": res_primal_history,
        "res_fixed_point_history": res_fixed_point_history
    }
