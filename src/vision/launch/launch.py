import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def handle_configuration(context, *args, **kwargs):
    # 1. Locate the configuration directory
    default_config_path = PathJoinSubstitution([FindPackageShare('vision'), 'config']).perform(context)
    user_cfg_dir = LaunchConfiguration('vision_config_path').perform(context)
    
    config_path = default_config_path  
    if user_cfg_dir and user_cfg_dir.strip():
        cand = user_cfg_dir.rstrip('/')
        if os.path.exists(os.path.join(cand, 'vision.yaml')):
            config_path = cand
        else:
            print(f"[vision launch] warning: {cand}/vision.yaml not found, falling back to {default_config_path}")
    
    # 2. Set file paths for the vision_node Init()
    # These match the 'Init(cfg_template, cfg_local)' signature in vision_node.cpp
    config_file = os.path.join(config_path, 'vision.yaml')
    config_local_file = os.path.join(config_path, 'vision_local.yaml')

    # Fallback if local config doesn't exist
    if not os.path.exists(config_local_file):
        config_local_file = config_file

    return [
        Node(
            package='vision',
            executable='vision_node',
            name='vision_node',
            output='screen',
            # Pass YAML paths as positional arguments (argv[1], argv[2])
            arguments=[config_file, config_local_file],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock (Webots/Isaac) if true'
        ),
        DeclareLaunchArgument(
            'vision_config_path',
            default_value='',
            description='Directory containing vision.yaml (empty uses package default)'
        ),
        OpaqueFunction(function=handle_configuration)
    ])