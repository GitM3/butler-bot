{
  lib,
  buildRosPackage,
  ament-cmake,
  rclcpp,
  cv-bridge,
  sensor-msgs,
  std-msgs,
  tf2-ros,
}:

buildRosPackage {
  pname = "detector";
  version = "0.1.0";

  src = lib.cleanSource ./src/detector;
  buildType = "ament_cmake";

  buildInputs = [
    rclcpp
    cv-bridge
    sensor-msgs
    std-msgs
    tf2-ros
  ];

  nativeBuildInputs = [ ament-cmake ];

  meta = {
    description = "Local detector ROS 2 package";
    license = lib.licenses.mit;
    maintainers = [ "" ];
  };
}
