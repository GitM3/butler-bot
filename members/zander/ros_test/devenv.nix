{
  pkgs, # The Nix package set (derived from inputs.nixpkgs)
  lib, # Nixpkgs utility functions
  config, # devenv configuration values
  nixpkgs, # Direct access to the nixpkgs input
  nixpkgs-unstable,
  nix-ros-overlay, # Direct access to the overlay input
  nixgl, # Direct access to the nixgl input
  ...
}:

let
  # Helper for nixGL configuration
  isIntelX86Platform = pkgs.stdenv.system == "x86_64-linux";
  nixGL = import nixgl {
    inherit pkgs;
    enable32bits = isIntelX86Platform;
    enableIntelX86Extensions = isIntelX86Platform;
  };
  pkgs-unstable = import nixpkgs-unstable {
    system = pkgs.system;
    config.allowUnfree = true;
  };
in
{
  name = "bb_ros";
  cachix.pull = [ "ros" ]; # Pull pre-built ROS packages
  overlays = [
    nix-ros-overlay.overlays.default
  ];

  # --- Packages ---
  packages =
    with pkgs;
    [
      colcon # The ROS 2 build tool
      graphviz # Often needed for ROS visualization tools
      cairo # Dependency for some GUI libraries

      # C++ Build Tools
      cmake
      gcc
      pkg-config

      # Computer Vision and Media Libraries
      opencv4
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-ugly
      gst_all_1.gst-libav
      v4l-utils # Video4Linux utilities for webcam access

      # --- Select ONE nixGL variant based on your GPU ---
      nixGL.auto.nixGLDefault # Often works
      # nixGL.nixGLIntel
      # nixGL.auto.nixGLNvidia
      # ... other variants
    ]
    ++ (with pkgs.rosPackages.jazzy; [
      # --- ROS 2 Packages ---
      (buildEnv {
        name = "bb_ros"; # Name for this specific ROS package group
        paths = [
          # Core ROS libraries
          ros-core
          ament-cmake-core

          rviz2 # For visualization
          nav2-amcl # Navigation stack component
          slam-toolbox # SLAM algorithms
          tf2-ros # Transform library
          tf2-tools # TF debugging tools
          rqt-graph
          rqt-common-plugins # Useful RQT GUI tools
          rqt-tf-tree # RQT TF visualization

          # Camera and image processing
          image-transport
          cv-bridge
          sensor-msgs
          rqt-image-view # For viewing camera feeds
          std-msgs
          rclpy
        ];
      })
    ]);

}
