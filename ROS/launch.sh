# IF JETSON:
## First run:
colcon build --symlink-install --packages-select py_detr
sed -i '1c#!/home/jetson/.pyenv/versions/py310/bin/python' install/py_detr/lib/py_detr/rfdetr_node

## Launch:
ros2 run py_detr rfdetr_node --ros-args -p model_path:=/home/jetson/BUTLER_BOT/models/rf-detr-small.onnx -p finish_time:=1
ros2 run dynamixel_sdk_examples realsense_pitch --ros-args --params-file /home/jetson/BUTLER_BOT/ROS/src/dynamix/realsense_pitch/dynamixel_sdk_examples/config/realsense_pitch.yaml

## Save homepoint
ros2 service call /save_homepoint std_srvs/srv/Trigger
