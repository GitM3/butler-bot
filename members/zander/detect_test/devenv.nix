{
  pkgs, # The Nix package set (derived from inputs.nixpkgs)
  lib, # Nixpkgs utility functions
  config, # devenv configuration values
  nixpkgs-unstable,
  ...
}:

let
  pkgs-unstable = import nixpkgs-unstable {
    system = pkgs.system;
    config.allowUnfree = true;
  };
in
{
  name = "bottleButler Bot";
  env = {
    # Prefer Wayland but allow X11 fallback if needed
    GDK_BACKEND = "wayland,x11";
    QT_QPA_PLATFORM = "wayland;xcb";
  };

  languages = {
    python = {
      enable = true;
      venv.enable = true;
      venv.requirements = ''
        pip
        opencv-python
        trackers
        roboflow==1.2.10
        idna==3.7
      '';
    };
    cplusplus.enable = true;
    c.enable = true;
  };
  packages = with pkgs; [
    (python313.withPackages (
      ps: with ps; [
        numpy
        transformers
        pillow
        timm
        requests
        pip
        virtualenv
        matplotlib
        #torch-bin
        torchvision
        (opencv4.override { enableGtk2 = true; })
      ]
    ))
    pyenv

    # ---- multimedia stack (libs) ----
    opencv # provides the native libs OpenCV wants
    ffmpeg

    # ---- GUI backends for HighGUI ----
    gtk3
    gtk2
    qt6.qtbase
    qt6.qtwayland
    xorg.xorgproto
    xorg.libX11
    xorg.libXext
    xorg.libSM
    xorg.libICE
    xwayland
    xorg.xhost
    wayland

    # ---- Camera / GStreamer stack ----
    v4l-utils
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-ugly
    gst_all_1.gst-libav
    gst_all_1.gst-vaapi

    #pkgs-unstable.claude-code
  ];
  # scripts."install-detr".exec = ''
  #   2. pip install "rfdetr @ git+https://github.com/roboflow/rf-detr.git@fix_nontype_in_optimize_call"

  # '';
}
