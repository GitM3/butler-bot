// Copyright 2025 Shimpei Sasaki
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

#ifndef REALSENSE_PITCH_HPP_
#define REALSENSE_PITCH_HPP_

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "std_msgs/msg/float64.hpp"
#include "tf2_ros/transform_broadcaster.h"


class ReadWriteNode : public rclcpp::Node
{
public:
  ReadWriteNode();
  ~ReadWriteNode() override;

private:
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr set_position_subscriber_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr present_angle_publisher_;
  rclcpp::TimerBase::SharedPtr present_angle_timer_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  uint8_t target_id_;
  int32_t min_goal_position_;
  int32_t max_goal_position_;
  double present_angle_publish_hz_;
  bool has_recent_goal_;
  int32_t present_position_;
  std::string parent_frame_;
  std::string child_frame_;
  double frame_offset_x_;
  double frame_offset_y_;
  double frame_offset_z_;
};

#endif  // REALSENSE_PITCH_HPP_
