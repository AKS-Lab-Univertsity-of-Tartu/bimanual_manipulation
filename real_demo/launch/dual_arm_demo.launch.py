import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Declare launch arguments with defaults
    use_config_file_arg = DeclareLaunchArgument('use_config_file', default_value='false')
    use_hardware_arg = DeclareLaunchArgument('use_hardware', default_value='false')
    record_data_arg = DeclareLaunchArgument('record_data', default_value='false')
    num_batch_arg = DeclareLaunchArgument('num_batch', default_value='500')
    num_steps_arg = DeclareLaunchArgument('num_steps', default_value='15')
    maxiter_cem_arg = DeclareLaunchArgument('maxiter_cem', default_value='1')
    maxiter_projection_arg = DeclareLaunchArgument('maxiter_projection', default_value='5')
    w_pos_arg = DeclareLaunchArgument('w_pos', default_value='3.0')
    w_rot_arg = DeclareLaunchArgument('w_rot', default_value='0.5')
    w_col_arg = DeclareLaunchArgument('w_col', default_value='500.0')
    num_elite_arg = DeclareLaunchArgument('num_elite', default_value='0.05')
    timestep_arg = DeclareLaunchArgument('timestep', default_value='0.1')
    position_threshold_arg = DeclareLaunchArgument('position_threshold', default_value='0.06')
    rotation_threshold_arg = DeclareLaunchArgument('rotation_threshold', default_value='0.1')

    # Launch configurations to fetch launch args
    use_config_file = LaunchConfiguration('use_config_file')
    use_hardware = LaunchConfiguration('use_hardware')
    record_data = LaunchConfiguration('record_data')
    num_batch = LaunchConfiguration('num_batch')
    num_steps = LaunchConfiguration('num_steps')
    maxiter_cem = LaunchConfiguration('maxiter_cem')
    maxiter_projection = LaunchConfiguration('maxiter_projection')
    w_pos = LaunchConfiguration('w_pos')
    w_rot = LaunchConfiguration('w_rot')
    w_col = LaunchConfiguration('w_col')
    num_elite = LaunchConfiguration('num_elite')
    timestep = LaunchConfiguration('timestep')
    position_threshold = LaunchConfiguration('position_threshold')
    rotation_threshold = LaunchConfiguration('rotation_threshold')

    # params = os.path.join(
    #         get_package_share_directory('real_demo'),
    #         'config',
    #         'planner_parameters.yaml'
    #     )
    
    params = {
        'use_hardware': use_hardware,
        'record_data': record_data,
        'num_batch': num_batch,
        'num_steps': num_steps,
        'maxiter_cem': maxiter_cem,
        'maxiter_projection': maxiter_projection,
        'w_pos': w_pos,
        'w_rot': w_rot,
        'w_col': w_col,
        'num_elite': num_elite,
        'timestep': timestep,
        'position_threshold': position_threshold,
        'rotation_threshold': rotation_threshold,
    }

    return LaunchDescription([
        use_config_file_arg,
        use_hardware_arg,
        record_data_arg,
        num_batch_arg,
        num_steps_arg,
        maxiter_cem_arg,
        maxiter_projection_arg,
        w_pos_arg,
        w_rot_arg,
        w_col_arg,
        num_elite_arg,
        timestep_arg,
        position_threshold_arg,
        rotation_threshold_arg,
        Node(
            package='real_demo',
            executable='dual_arm_demo',
            name='planner',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
            parameters=[params],
        )
    ])