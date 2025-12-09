cmake_minimum_required(VERSION 3.16)
project(rfdetr_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# librealsense2
find_package(realsense2 REQUIRED)

# ONNX Runtime (simple discovery)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIB NAMES onnxruntime)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(
    FATAL_ERROR
      "onnxruntime headers or library not found. Set CMAKE_INCLUDE_PATH/CMAKE_LIBRARY_PATH or ensure onnxruntime is installed."
  )
endif()

# Optional: ONNX Runtime CUDA provider (for Jetson/Ubuntu with CUDA)
find_library(ONNXRUNTIME_PROVIDERS_SHARED NAMES onnxruntime_providers_shared)
find_library(ONNXRUNTIME_CUDA_LIB NAMES onnxruntime_providers_cuda)
set(HAVE_ORT_CUDA FALSE)
if(ONNXRUNTIME_PROVIDERS_SHARED AND ONNXRUNTIME_CUDA_LIB)
  set(HAVE_ORT_CUDA TRUE)
  add_compile_definitions(ENABLE_ORT_CUDA)
  message(STATUS "Building with ONNX Runtime CUDA provider support")
else()
  message(STATUS "ONNX Runtime CUDA provider not found; building CPU-only")
endif()

add_executable(rfdetr_node src/rfdetr_node.cpp)
target_include_directories(rfdetr_node PRIVATE ${OpenCV_INCLUDE_DIRS}
                                               ${ONNXRUNTIME_INCLUDE_DIR})

ament_target_dependencies(rfdetr_node rclcpp sensor_msgs std_msgs cv_bridge)

target_link_libraries(rfdetr_node ${OpenCV_LIBS} ${ONNXRUNTIME_LIB}
                      realsense2::realsense2)

if(HAVE_ORT_CUDA)
  target_link_libraries(rfdetr_node ${ONNXRUNTIME_PROVIDERS_SHARED}
                        ${ONNXRUNTIME_CUDA_LIB})
endif()

install(TARGETS rfdetr_node DESTINATION lib/${PROJECT_NAME})

ament_package()
