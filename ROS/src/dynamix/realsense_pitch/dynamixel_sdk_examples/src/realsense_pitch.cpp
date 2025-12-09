// Copyright 2025 Shimpei Sasaki
// This file manages the control of the Dynamixel motor and facilitates communication with ROS 2.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "dynamixel_sdk/dynamixel_sdk.h"
#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include "realsense_pitch.hpp"

// Control table addresses for AX-12A (Protocol 1.0) - These addresses are used to access motor parameters.
// Note: The AX-12A does not have an Operating Mode register.
#define ADDR_TORQUE_ENABLE 24    // AX-12A Torque Enable Address (1 Byte)
#define ADDR_GOAL_POSITION 30    // AX-12A Goal Position Address (2 Bytes)
#define ADDR_PRESENT_POSITION 36 // AX-12A Present Position Address (2 Bytes)

// Protocol version
#define PROTOCOL_VERSION 1.0     // AX-12A uses Protocol 1.0

// Default settings for motor communication.
#define BAUDRATE 1000000    // Common initial baudrate for AX-12A (change if necessary)
#define DEVICE_NAME "/dev/ttyUSB0"  // [Linux]: "/dev/ttyUSB*", [Windows]: "COM*"

dynamixel::PortHandler * portHandler;
dynamixel::PacketHandler * packetHandler;

uint8_t dxl_error = 0;
int dxl_comm_result = COMM_TX_FAIL;

namespace
{
constexpr int32_t kDefaultMinGoalPosition = 205;
constexpr int32_t kDefaultMaxGoalPosition = 546;
constexpr int32_t kDefaultFixedDxlId = 5;
constexpr double kDegToRad = M_PI / 180.0;
}

void setupDynamixel(uint8_t dxl_id);

using namespace std::chrono_literals;

ReadWriteNode::ReadWriteNode()
: Node("realsense_pitch_node"),
  target_id_(static_cast<uint8_t>(kDefaultFixedDxlId)),
  min_goal_position_(kDefaultMinGoalPosition),
  max_goal_position_(kDefaultMaxGoalPosition),
  present_angle_publish_hz_(30.0),
  has_recent_goal_(false),
  present_position_(0)
{
  RCLCPP_INFO(this->get_logger(), "Run read write node");

  this->declare_parameter("qos_depth", 5);
  int8_t qos_depth = 0;
  this->get_parameter("qos_depth", qos_depth);

  this->declare_parameter<int64_t>("min_goal_position", kDefaultMinGoalPosition);
  this->declare_parameter<int64_t>("max_goal_position", kDefaultMaxGoalPosition);
  this->declare_parameter<int64_t>("fixed_dxl_id", kDefaultFixedDxlId);
  parent_frame_ = this->declare_parameter<std::string>("parent_frame", "base_link");
  child_frame_ = this->declare_parameter<std::string>("child_frame", "camera_pitch");
  frame_offset_x_ = this->declare_parameter<double>("frame_offset_x", 0.042);
  frame_offset_y_ = this->declare_parameter<double>("frame_offset_y", 0.0);
  frame_offset_z_ = this->declare_parameter<double>("frame_offset_z", 0.260);
  present_angle_publish_hz_ = this->declare_parameter<double>("present_angle_publish_hz", 30.0);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  min_goal_position_ = static_cast<int32_t>(this->get_parameter("min_goal_position").as_int());
  max_goal_position_ = static_cast<int32_t>(this->get_parameter("max_goal_position").as_int());
  if (min_goal_position_ >= max_goal_position_) {
    RCLCPP_WARN(
      this->get_logger(),
      "min_goal_position (%d) is not less than max_goal_position (%d). Using defaults.",
      min_goal_position_,
      max_goal_position_);
    min_goal_position_ = kDefaultMinGoalPosition;
    max_goal_position_ = kDefaultMaxGoalPosition;
  }

  const auto requested_id = this->get_parameter("fixed_dxl_id").as_int();
  const int clamped_id = std::clamp<int>(static_cast<int>(requested_id), 0, 253);
  if (static_cast<int64_t>(clamped_id) != requested_id) {
    RCLCPP_WARN(
      this->get_logger(),
      "fixed_dxl_id (%ld) is out of range [0, 253]. Clamping to %d.",
      requested_id,
      clamped_id);
  }
  target_id_ = static_cast<uint8_t>(clamped_id);

  if (present_angle_publish_hz_ <= 0.0) {
    RCLCPP_WARN(
      this->get_logger(),
      "present_angle_publish_hz (%f) must be greater than 0. Using default %.1f.",
      present_angle_publish_hz_,
      30.0);
    present_angle_publish_hz_ = 30.0;
  }

  const auto qos_profile =
    rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable().durability_volatile();

  set_position_subscriber_ = this->create_subscription<std_msgs::msg::Float64>(
    "set_position",
    qos_profile,
    [this](const std_msgs::msg::Float64::SharedPtr msg) -> void
    {
      uint8_t dxl_error = 0;

    const double requested_angle_deg = msg->data;
    const double raw_goal = static_cast<double>(min_goal_position_) + requested_angle_deg * 1023.0 / 300.0;
    int32_t goal_position_ticks = static_cast<int32_t>(std::lround(raw_goal));
    goal_position_ticks = std::clamp(goal_position_ticks, min_goal_position_, max_goal_position_);

      // The position value for AX-12A is represented as 2-byte data (uint16_t).
      uint16_t goal_position_uint16 = static_cast<uint16_t>(goal_position_ticks);

      // Write the Goal Position (2 bytes in length) to the motor.
      // Use write2ByteTxRx for AX-12A
      dxl_comm_result = packetHandler->write2ByteTxRx(
        portHandler,
        target_id_,
        ADDR_GOAL_POSITION,
        goal_position_uint16,
        &dxl_error
      );

      if (dxl_comm_result != COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getTxRxResult(dxl_comm_result));
      } else if (dxl_error != 0) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getRxPacketError(dxl_error));
      } else {
        has_recent_goal_ = true;
        RCLCPP_INFO(
          this->get_logger(),
          "Set [ID: %u] [Goal Position: %d] (requested angle: %.1f deg)",
          static_cast<unsigned int>(target_id_),
          goal_position_ticks,
          requested_angle_deg
        );
      }
    }
  );

  present_angle_publisher_ = this->create_publisher<std_msgs::msg::Float64>("present_angle", qos_profile);

  const auto publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / present_angle_publish_hz_));

  present_angle_timer_ = this->create_wall_timer(
    publish_period,
    [this]() {
      uint16_t read_data_uint16 = 0;
      uint8_t local_error = 0;
      int local_comm_result = packetHandler->read2ByteTxRx(
        portHandler,
        target_id_,
        ADDR_PRESENT_POSITION,
        &read_data_uint16,
        &local_error
      );

      if (local_comm_result != COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getTxRxResult(local_comm_result));
        return;
      }

      if (local_error != 0) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getRxPacketError(local_error));
        return;
      }

      present_position_ = static_cast<int32_t>(read_data_uint16);

      std_msgs::msg::Float64 angle_msg;
      // Convert the internal position to degrees using the same offset and scale as in set_position.
      // degree = (present_position - OFFSET) * 300 / 1023
      double angle = static_cast<double>(present_position_ - min_goal_position_) * 300.0 / 1023.0;
      if (std::abs(angle) <= 0.5) {
        angle = 0.0;
      }
      angle_msg.data = angle;
      present_angle_publisher_->publish(angle_msg);

      if (tf_broadcaster_ != nullptr) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->now();
        tf_msg.header.frame_id = parent_frame_;
        tf_msg.child_frame_id = child_frame_;
  tf_msg.transform.translation.x = frame_offset_x_;
  tf_msg.transform.translation.y = frame_offset_y_;
  tf_msg.transform.translation.z = frame_offset_z_;

        tf2::Quaternion q;
        q.setRPY(0.0, -angle * kDegToRad, 0.0);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(tf_msg);
      }
    }
  );

  setupDynamixel(target_id_);
}

ReadWriteNode::~ReadWriteNode()
{
  if (packetHandler != nullptr && portHandler != nullptr) {
    packetHandler->write1ByteTxRx(
      portHandler,
      target_id_,
      ADDR_TORQUE_ENABLE,
      0,
      &dxl_error
    );
  }
}

void setupDynamixel(uint8_t dxl_id)
{
  // No need to set the Operating Mode for AX-12A (it is always in joint/wheel mode).
  // Enable the torque of the DYNAMIXEL motor.
  dxl_comm_result = packetHandler->write1ByteTxRx(
    portHandler,
    dxl_id,
    ADDR_TORQUE_ENABLE,
    1,
    &dxl_error
  );

  if (dxl_comm_result != COMM_SUCCESS) {
    RCLCPP_ERROR(rclcpp::get_logger("realsense_pitch_node"), "Failed to enable torque.");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("realsense_pitch_node"), "Succeeded to enable torque.");
  }
}

int main(int argc, char * argv[])
{
  portHandler = dynamixel::PortHandler::getPortHandler(DEVICE_NAME);
  packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  // Open the serial port for communication.
  dxl_comm_result = portHandler->openPort();
  if (dxl_comm_result == false) {
    RCLCPP_ERROR(rclcpp::get_logger("realsense_pitch_node"), "Failed to open the port!");
    return -1;
  } else {
    RCLCPP_INFO(rclcpp::get_logger("realsense_pitch_node"), "Succeeded to open the port.");
  }

  // Set the baud rate of the serial port (use the DYNAMIXEL baud rate).
  dxl_comm_result = portHandler->setBaudRate(BAUDRATE);
  if (dxl_comm_result == false) {
    RCLCPP_ERROR(rclcpp::get_logger("realsense_pitch_node"), "Failed to set the baudrate!");
    return -1;
  } else {
    RCLCPP_INFO(rclcpp::get_logger("realsense_pitch_node"), "Succeeded to set the baudrate.");
  }

  rclcpp::init(argc, argv);

  auto readwritenode = std::make_shared<ReadWriteNode>();
  rclcpp::spin(readwritenode);
  rclcpp::shutdown();

  return 0;
}