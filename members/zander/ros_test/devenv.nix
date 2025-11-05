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
  localOverlay = import ./overlay.nix;
  pkgs-unstable = import nixpkgs-unstable {
    system = pkgs.system;
    config.allowUnfree = true;
  };
in
{
  name = "bb_ros";
  cachix = {
    enable = true;
    pull = [ "ros" ]; # Pull pre-built ROS packages
  };
  overlays = [
    nix-ros-overlay.overlays.default
    #localOverlay
  ];

  languages = {
    python = {
      enable = true;
      #package = pkgs.python313;
      venv.enable = true;
      # Using pip/venv since ROS can use this instead of nix store.
      venv.requirements = ''
        pip
        opencv-python
        trackers
        supervision==0.27.0rc1
        roboflow==1.2.10
        idna==3.7
        numpy
        transformers==4.57.0
        timm
        matplotlib
        onnxruntime>1.17
        onnx==1.19.0
        onnxscript
        onnxsim==0.4.36
        torchvision
        torch==2.8.0
        git+https://github.com/roboflow/rf-detr.git
        lazyros
      '';
    };
    cplusplus.enable = true;
    c.enable = true;
  };
  # --- Packages ---
  packages =
    with pkgs;
    [
      pkgs-unstable.codex
      colcon # The ROS 2 build tool
      graphviz # Often needed for ROS visualization tools
      cairo # Dependency for some GUI libraries

      # C++ Build Tools
      cmake
      gcc
      pkg-config
      onnxruntime

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
      # nixGL.auto.nixGLDefault # Often works
      nixGL.nixGLIntel
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
          python-cmake-module

          rviz2 # For visualization
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
          librealsense2
          realsense2-camera
        ];
      })
    ]);

  enterShell = ''
    export PYTHONPATH=$PWD/install/py_detr/lib/python3.12/site-packages:$PYTHONPATH
    export PYTHONPATH=$PWD/.devenv/state/venv/lib/python3.12/site-packages:$PYTHONPATH
    # export CMAKE_PREFIX_PATH=${pkgs.rosPackages.jazzy.cv-bridge}/share:$CMAKE_PREFIX_PATH

    if [[ ! $DIRENV_IN_ENVRC ]]; then
        eval "$(${pkgs.python3Packages.argcomplete}/bin/register-python-argcomplete ros2)"
        eval "$(${pkgs.python3Packages.argcomplete}/bin/register-python-argcomplete colcon)"
    fi

  '';
}
