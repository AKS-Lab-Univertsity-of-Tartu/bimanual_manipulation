# Dual Arm Manipulation

This repository contains ROS2 Humble package for dual arm manipulation with the sampling based planner.

## Setup

* Ubuntu 24.04.2 LTS
* ROS2 Humble
* UR5e robots 
* NVIDIA GeForce RTX™ 3090

## Package Structure
* ***___data/___:***  Folder to which the setup info and trajectory are saved.
* ***___ur5e_hande_mjx/___:***  The MuJoCo model description.
* ***___sampling_based_planner/___:***  Sampling based planner.
* ***___launch/___ :***  Launch files for dual_arm_demo and visualizer.
* ***___config/___ :***  Config files with parameters for dual_arm_demo and visualizer.
* ***___dual_arm_demo.py___ :***  The code for online planning using Model Predictive Control strategy (has both hardware and non-hardware).
* ***___visualizer.py___ :***  The code for visualizing current MuJoCo environment and/or recorded trajectory from ___/data___ folder.

## Installation Guide

Create colcon workspace and clone the repository:
```bash
$ mkdir -p ~/colcon_ws/src
$ cd ~/colcon_ws/src
$ git clone -b issue_22_pass_cube https://github.com/alinjar1996/manipulator_mujoco.git
```

Pull submodules and build workspace:
```bash
$ cd ~/colcon_ws/src/manipulator_mujoco
$ git submodule update --init --recursive
$ cd ~/colcon_ws
$ colcon build
```

Create virtual python environment and install requirements:
```bash
$ python3 -m venv ~/{name}_env
$ source ~/{name}_env/bin/activate
$ cd ~/colcon_ws/src/manipulator_mujoco
$ pip install -r requirements.txt
$ deactivate
```

Source the workspace and export virtual environment path (has to be done each time a new terminal is opened):
```bash
$ cd ~/colcon_ws
$ source install/setup.bash
$ export PYTHONPATH=/path/to/env/{name}/lib/python3.12/site-packages:$PYTHONPATH
```

## Run

To check what arguments are available to set from command line:
```bash
ros2 launch real_demo dual_arm_demo.launch.py --show-arguments
```

Run planner:
```bash
$ ros2 launch real_demo dual_arm_demo.launch.py [arg_name:=arg_value]
```



