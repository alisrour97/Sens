#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='trajectory_publisher',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            parameters=[os.path.join(
                get_package_share_directory('trajectory_publisher'),
                'conf', 'param.yaml')],
            output='screen'),
    ])

