cmake_minimum_required(VERSION 3.5)
project(dynamixel_sdk_examples)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)
find_package(rcutils REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(dynamixel_sdk_custom_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
include_directories(include)
add_executable(realsense_pitch src/realsense_pitch.cpp)
target_include_directories(realsense_pitch PRIVATE include)
ament_target_dependencies(realsense_pitch
  dynamixel_sdk
  dynamixel_sdk_custom_interfaces
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
)

add_executable(read_write_node src/read_write_node.cpp)
target_include_directories(read_write_node PRIVATE include)
ament_target_dependencies(read_write_node
  dynamixel_sdk
  dynamixel_sdk_custom_interfaces
  rclcpp
  rcutils
)

# Install
install(TARGETS
  read_write_node
  realsense_pitch
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/read_write_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
