from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription([
        Node(package="py_detr", executable="video_node", name="video_node"),
        Node(package="py_detr", executable="rfdetr_node", name="rfdetr_node"),
        Node(package="py_detr", executable="tracker_node", name="tracker_node"),
    ])

