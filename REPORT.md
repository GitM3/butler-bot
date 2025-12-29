# RF DETR Node – Object Detector and Tracker

## Abstract Overview
`RFDetrNode` is a ROS 2 node that couples an ONNX RF-DETR detector with RealSense RGB-D sensing, depth-based contour refinement, and a Kalman-aided state machine to command a pan/tilt head and publish target TF frames. The implementation integrates perception, temporal filtering, control, and state transitions in a single timer-driven loop.

## System Architecture
- **Perception stack**: `RFDETR_ONNX` wraps ONNX Runtime (CUDA/CPU) and handles RGB preprocessing/postprocessing to yield class-filtered boxes. Depth utilities build adaptive masks, extract contours, deproject pixels to 3D, and smooth bboxes with an 8D Kalman filter.
- **Control stack**: Pixel-space error drives yaw/pitch setpoints with deadbands and clamping; a search oscillation runs when targets are missing. Annotated RGB, state flags, and TF (`target_frame`, `home_point`) are published for downstream consumers.
- **Coordination**: A discrete state machine sequences perception and control policies across detection, approach, tracking, finish, and search behaviors, using debounced evidence from both detector and depth contour.

## Perception Pipeline
1. **Capture & alignment**: RealSense pipeline streams aligned color/depth (or bag playback). Depth is optionally filtered.
2. **Detection**: RGB → `predict()` → scores/labels/boxes for cup/bottle/wine-glass IDs only. Depth is sampled at each box center.
3. **ROI formation**: Depending on state, ROI derives from current detection, Kalman prediction, prior contour center, or full-frame fallback. `expand_bbox`, `bbox_from_center`, and `full_frame_bbox` manage ROI geometry.
4. **Depth masking & contouring**: `build_depth_mask` uses percentile depth around ROI plus margin and morphology to isolate near-field structure. `find_depth_contour` selects the largest valid contour and returns its centroid and outline.
5. **Tracking filter**: Measurements (contour-derived bbox) correct the Kalman filter; predictions bridge detector/contour dropouts. Bboxes are clamped to image bounds before use.
6. **3D projection**: Selected target pixel+depth are deprojected via camera intrinsics to meters and published as `target_frame` TF.

## Depth Mask and Contour Selection (Detailed)
- **Base mask**: Start with depth > 0 and ≤ `depth_mask_threshold_mm`, producing a broad binary foreground over valid ranges.
- **ROI-driven dynamic window**:
  - If an ROI is available, crop depth to ROI, gather valid depths, and compute a lower bound `d_min` via `depth_mask_percentile`. This biases toward the closest material within the ROI, assuming the target is the nearest object in that region.
  - Accept depths in `[d_min, d_min + depth_mask_margin_mm]`, yielding a narrow band around the closest surface. This suppresses background at larger depths even if the detector box is loose.
  - Apply morphological close → dilate → close with an elliptical kernel (`depth_kernel_size`, enforced odd) to fill pinholes and smooth mask edges.
  - Paste the cleaned ROI mask back into a full-frame mask located at the ROI coordinates.
- **Fallback window**: If ROI is missing or empty, compute `d_min` over a central window (middle half of the image) or over all valid depths. Use a minimum margin of 5 mm and the same morphology to yield a scene-wide near-depth mask.
- **Contour extraction (`find_depth_contour`)**:
  - Optionally crop the mask to the ROI again for search efficiency.
  - Run `cv2.findContours` on the binary mask, select the largest contour by area, and discard if below `depth_min_contour_area`.
  - Compute centroid via image moments; translate to full-frame coordinates if cropped.
  - Return centroid and contour; these feed Kalman correction, servo aiming, and state gating.
- **Usage across states**:
  - **APPROACH**: Mask is built around detector/KF ROI to confirm and tighten the target via depth; a stable contour can trigger `TRACK` or a quick finish.
  - **TRACK**: When the detector drops, masking around predicted/previous centers keeps lock. Stable contours sustain tracking; loss pushes the machine toward `SEARCH`.
  - **SEARCH**: No contouring; oscillation continues until a detector-driven ROI restarts the mask/contour process.

## State Machine
- **States**: `DETECT`, `APPROACH`, `TRACK`, `FINISH`, `SEARCH` (published on `/state/detector`; `/state/lost` toggles presence).
- **Debounce**: Detection and contour stability are counted (`DETECT_YES/NO_THRESH`, `CONTOUR_YES/NO_THRESH`) to avoid chattering.
- **Transitions**:
  - `DETECT`: Stable detection within `depth_threshold` → `APPROACH`; prolonged loss → `SEARCH`; stable detection asserts `found`.
  - `APPROACH`: ROI from detection/KF; seek depth contour. Stable contour + high pitch → `TRACK`; stable contour + close depth may `FINISH` early. Loss before confirmation → `SEARCH` with KF reset.
  - `TRACK`: If detector reappears while pitched down, return to `DETECT`; else follow contour/KF. Stable contour for `finish_time` → `FINISH`; sustained contour loss → `SEARCH` then reset.
  - `FINISH`: Optionally publish home TF; any new stable detection drops back to `DETECT`.
  - `SEARCH`: Servo oscillation until detection stabilizes, then `DETECT`.
- **Quick finish**: `finish_quick` shortcuts to `FINISH` when pitch is high and depth < 0.2 m.

## Servo Control and Targeting
- **Error mapping**: Preferred target point is contour centroid; fallback is detection center. Pixel error vs. image center outside deadbands (`x_max`, `y_max`) produces yaw/pitch adjustments scaled to image size and clamped to `[pitch_min, pitch_max]`.
- **Actuation policy**: Commands publish only on change (`/set_position`, `/set_yaw`). In `SEARCH`, `oscillate_servo` sweeps pitch to aid reacquisition.

## Frames, Services, and Integration
- **Frames**: Static optical frame under `camera_pitch`; dynamic `target_frame` at deprojected target; `home_point` saved in `map` via TF lookup on startup or `save_homepoint` service.
- **Navigation hook**: `set_navigation_pause` client is prepared (commented in transitions) for coordination with a navigation stack.
- **Outputs**: Annotated images (optional), TF broadcasts, state/lost topics, and target point TF for planners or controllers.

## Parameters and Timing
- **Model/targets**: `model_path`, `target_class` (subset of COCO).  
- **Depth/contour**: `depth_mask_threshold_mm`, `depth_mask_percentile`, `depth_mask_margin_mm`, `depth_kernel_size`, `depth_min_contour_area`, `track_window_px`, `search_bbox_margin`, `depth_threshold`.  
- **Control/timing**: `frame_rate`, `finish_time`, `pitch_step`, servo limits, debounce thresholds.  
- **Debug/runtime**: `annotate`, `debug_time`, `pub_return`, `bag_path`. Timing logs (every 15 frames) break down capture, filtering, detection, state logic, and remainder.

## Execution Cycle (per timer tick)
1. Acquire aligned RGB/depth.
2. Detect targets and sample depth at box centers.
3. Update detection/contour stability counters.
4. Advance state machine (set ROI, build mask, contour search, KF predict/correct, transitions).
5. Select target point, update servos if not searching.
6. Deproject to 3D, publish TF/annotations/state flags, optionally log timings.

This architecture layers detector outputs, depth structure, and temporal filtering behind a gated state machine, yielding robust target lock and actionable 3D pose estimates despite intermittent detections or depth noise. 
