# IF JETSON:
ros2 run py_detr rfdetr_node --ros-args -p model_path:=/home/jetson/BUTLER_BOT/members/zander/models/rf-detr-small.onnx
ros2 launch realsense2_camera rs_launch.py
ros2 run dynamixel_sdk_examples realsense_pitch --ros-args --params-file /home/jetson/BUTLER_BOT/dynamixel_sdk_examples/config/realsense_pitch.yaml
