import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():

    # Declare launch arguments
    use_config_file_arg = DeclareLaunchArgument('use_config_file', default_value='false')
    use_hardware_arg = DeclareLaunchArgument('use_hardware', default_value='false')
    record_data_arg = DeclareLaunchArgument('record_data', default_value='false')
    playback_arg = DeclareLaunchArgument('playback', default_value='true')
    folder_arg = DeclareLaunchArgument('folder', default_value='planner')
    idx_arg = DeclareLaunchArgument('idx', default_value='2')

    # LaunchConfigurations to use in Node parameters
    use_config_file = LaunchConfiguration('use_config_file')
    use_hardware = LaunchConfiguration('use_hardware')
    record_data = LaunchConfiguration('record_data')
    playback = LaunchConfiguration('playback')
    folder = LaunchConfiguration('folder')
    idx = LaunchConfiguration('idx')

    # params = os.path.join(
    #     get_package_share_directory('real_demo'),
    #     'config',
    #     'visualizer_parameters.yaml'
    # )

    params = {
        'use_hardware': use_hardware,
        'record_data': record_data,
        'playback': playback,
        'folder': folder,
        'idx': idx
    }

    return LaunchDescription([
        use_config_file_arg,
        use_hardware_arg,
        record_data_arg,
        playback_arg,
        folder_arg,
        idx_arg,
        Node(
            package='real_demo',
            executable='visualizer',
            name='visualizer',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
            parameters=[params],
        )
    ])