## Ellipse Tracking Strategy

- **Implemented approach:** `rf_detr_node.py` now instantiates a constant-velocity Kalman filter (state `[cx, cy, vx, vy]`) whenever an ellipse observation is available during `ELLIPSE_SEARCH`, and uses it to predict the next center while in `ELLIPSE_TRACK`. That prediction defines a tight ROI for `_detect_ellipse_in_bbox`, making each frame’s ellipse search faster and more robust.
- **Dynamic assumptions captured in the filter:**  
  - Motion is smooth between frames, so the state transition matrix embeds `cx += vx * dt` and `cy += vy * dt` with small process noise tuned for typical robot speeds.  
  - The ellipse moves mostly in the image plane; thus 2‑D position plus velocity suffices.  
  - Short observation gaps occur; the filter keeps propagating predictions (with process noise `ellipse_process_var`) so the ROI still follows the likely location for a brief period.  
  - Depth still comes directly from the depth image lookup, but the Kalman tracker could be extended with another dimension if depth-motion coupling becomes important.
- **Operational notes:** Resetting the tracker whenever we fall back to `DETECT` prevents stale predictions, and the adaptive ROI size (derived from the last bounding box) keeps the search window proportional to the bottle bottom while clamping it between 60–200 px for speed.

## `rf_detr_node.py` Overview

- **Key data structures**
  - `RFDETR_ONNX`: wraps ONNX Runtime session, handles preprocessing/post-processing, and returns `scores/labels/boxes`.
  - `State` enum: `DETECT`, `ELLIPSE_SEARCH`, `ELLIPSE_TRACK` stored in `self.state`.
  - Detection dictionaries: each contains `class_id`, `class_name`, `score`, `bbox`, `center`, and `depth`; lists of these are published via `Float64MultiArray`.
  - Ellipse tracker: `self.kf_state`, `self.kf_cov`, `self.kf_last_time`, and `self.ellipse_window_px` implement the Kalman filter and adaptive ROI sizing.
  - Timing/depth cache: `self.last_seen_object_time`, `self.last_seen_ellipse_time`, `self.depth_array`, and `self.depth_scale` enable state transitions and depth metrics.

- **Function responsibilities**
  - `callback`: synchronizes RGB/depth, runs `detect_objects`, drives the state machine, and publishes annotated frames plus detection arrays.
  - `detect_objects`: limits RF-DETR results to allowed COCO IDs, derives centers/depth via `get_depth_at`, and returns everything needed for downstream logic.
  - `find_ellipse`/`_detect_ellipse_in_bbox`: search for ellipses in a ROI; data feeds the Kalman tracker through `_ellipse_tracker_observe`.
  - `track_ellipse`: predicts the next center with `_ellipse_predict`, crops a search window, re-detects the ellipse, and updates the tracker for the next cycle.
  - `publish_detections`: flattens detection dictionaries so subscribers can parse `[class_id, cx, cy, depth]`.
  - `get_depth_at`: single depth lookup that handles uint16/float encodings.

- **Missing features / perf opportunities**
  - `track_ellipse` still re-runs contour extraction per frame; caching gradient images or using edge-aware templates could reduce computation.
  - Depth handling assumes a single scale; extending to dynamically read the camera’s scale (e.g., via `CameraInfo`) would improve robustness.
  - Publishing only the target class is straightforward (filter before `publish_detections`) but currently every allowed class is output.
  - `detect_objects` runs on full-resolution frames; adding optional downscaling or batching could increase FPS if the GPU is saturated.
  - The ellipse tracker ignores measurement confidence; incorporating ellipse fit residuals into the Kalman update could weight noisy observations better.
  - No persistence of detections when the model outputs nothing; a short-lived extrapolation of the last target detection before falling back to `DETECT` could keep the loop steady.
