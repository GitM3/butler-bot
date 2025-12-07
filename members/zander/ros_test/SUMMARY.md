## `src/py_detr/py_detr/rf_detr_node.py`

- ROS2 node that acquires aligned RGB/depth frames from an Intel RealSense pipeline, runs RF-DETR (ONNXRuntime on CUDA/CPU) to detect target COCO objects (`cup`, `bottle`, `wine glass`), and publishes annotated imagery plus `[class_id, cx, cy, depth]` arrays for every allowed detection.
- Depth-filtered detections gate servo control: the node debounces RF-DETR output (`DETECT_YES_THRESH=3`, `DETECT_NO_THRESH=5` frames) before commanding `/set_position` (pitch) and `/set_yaw` errors based on the offset between the image center and the chosen detection/ellipse centroid.
- Publishes several debug feeds (`/ellipse/find_mask`, `/ellipse/find_debug`, edges, bbox gate, masked edges, `/ellipse/track_debug`) to visualize intermediate stages of the ellipse pipeline and Kalman predictions.

## State Machine

- `State.DETECT` (default): waits for a stable target detection inside `depth_threshold` (600 mm). Once confirmed, transitions to `ELLIPSE_SEARCH` and resets the Kalman filter/ellipse counters.
- `State.ELLIPSE_SEARCH`: assumes the object is visible and searches for a valid ellipse within the detection gate.
  - If detections disappear before an ellipse is confirmed, it falls back to `DETECT`.
  - If detections vanish *after* an ellipse is confirmed, it moves to `ELLIPSE_TRACK`.
  - Each raw ellipse observation primes the Kalman filter for smoother major/minor axis and angle estimates.
- `State.ELLIPSE_TRACK`: object is lost but an ellipse is known. The node keeps predicting ellipse states via a 10D constant-velocity Kalman filter (`[cx, cy, major, minor, angle, v...]`) and tries to re-detect near the prediction using `fallback_ellipse_search`.
  - Regains `DETECT` when the object is detected again or the ellipse is missing for `ELLIPSE_NO_THRESH` frames.
- Each transition logs a reason so operators can correlate system responses with telemetry.

## Ellipse Detection & Tracking

- **Depth-based seeding:** Pixels within `depth_threshold` form `init_depth_mask`, dilated with a `mask_kernel_size`-sized elliptical kernel to expand the search region (`expanded_mask`). This mask limits where ellipse fitting runs and reduces background clutter.
- **Edge extraction:** The RGB image is masked by `expanded_mask`, converted to grayscale, blurred, and passed through Canny using configurable thresholds (`ellipse_canny_low/high`). The resulting edges are ANDed with the mask to create `edges_masked`.
- **Detection gate:** When a bounding box from RF-DETR is available, a 10 % margin is added around it to form `bdetect`, further restricting ellipse candidates to the vicinity of the target bottle/cup. All gate masks and edge maps are published for debugging.
- **Contour screening:** `cv2.findContours` runs on `edges_masked`. Contours with <5 points or `area < ellipse_min_contour_area` are ignored. The remaining contours are fit with `cv2.fitEllipse`, optionally filtered to lie inside the gated bbox. The ellipse with the largest area becomes the candidate and is drawn in green on the debug frame.
- **Kalman filter:** A `cv2.KalmanFilter(10,5)` tracks `(cx, cy, major_axis, minor_axis, angle)` with constant-velocity dynamics (the hidden state stores parameters plus their velocities). During `ELLIPSE_SEARCH`, every raw ellipse measurement updates the filter. In `ELLIPSE_TRACK`, predictions provide a `tracked_ellipse` even if the detector sees nothing, keeping the servo aim stable and supplying the ROI for `fallback_ellipse_search`.
- **Fallback search:** If the direct ellipse search fails during tracking, the node dilates masks around both the previous ellipse and the current Kalman prediction at multiple scales (1–4× `mask_kernel_size`) and reruns `find_ellipse`. This extends the lifetime of the track when the ellipse is partially occluded.
- **Outputs:** The currently tracked ellipse center drives the servo controller and is overlaid on `/camera/annotated`. Debug streams show both the raw detection (blue) and Kalman-stabilized ellipse (red) so tuning can focus on either measurement or tracking noise.

## Detailed Flow

- **Stability criteria:**
  - `object_detected_raw` toggles `detect_yes`/`detect_no`. An object is *stable* when `detect_yes >= DETECT_YES_THRESH` (3 frames) and *lost* when `detect_no >= DETECT_NO_THRESH` (5 frames). Counters reset each time the opposite condition occurs so brief dropouts do not flip states.
  - `ellipse_found_raw` uses the same debounce idea with `ellipse_yes`/`ellipse_no`. At least 3 consecutive ellipse frames confirm `ellipse_found_stable`; 5 consecutive misses declare `ellipse_lost_stable`.
- **State transition ordering inside `callback`:**
  1. Acquire frames, run `detect_objects`, annotate bounding boxes, and compute mask/ellipse candidates.
  2. Update detection counters, compute `object_detected_stable`/`object_lost_stable`.
  3. Update ellipse counters, compute `ellipse_found_stable`/`ellipse_lost_stable`.
  4. Switch over `self.state`:
     - `DETECT`: wait until both a stable detection exists and its depth is below `depth_threshold`. When true, transition to `ELLIPSE_SEARCH`, reset Kalman filter (`kf_reset`) and ellipse counters.
     - `ELLIPSE_SEARCH`: draw tentative ellipses for visualization. If detections drop before the ellipse stabilizes, fall back to `DETECT`. If the ellipse stabilizes but detections drop afterward, move to `ELLIPSE_TRACK` and remember the last ellipse. Every frame with a raw ellipse updates the Kalman filter to smooth axes/angle.
     - `ELLIPSE_TRACK`: immediately return to `DETECT` if detections become stable again. Otherwise, predict the ellipse via `kf_predict`; if no ellipse is detected, run `fallback_ellipse_search` using dilated masks built from `prev_ellipse` and the prediction. Lose the ellipse for 5 frames → `DETECT`. A found ellipse updates `prev_ellipse`, corrects the Kalman filter, and provides the aim point for servo control and `/ellipse/track_debug`.
  5. After state handling, servo commands use the best available centroid (`det_center` in `DETECT`/`ELLIPSE_SEARCH`, `tracked_ellipse[0]` while tracking) so the platform keeps aiming even when detections drop.
- **Ellipse pipeline nuances:**
  - `init_depth_mask` is purely depth-based; `ellipse_depth_margin` defines how far outside the depth slice the dilation may grow.
  - Bounding-box gating uses 10 % padding and applies via `cv2.rectangle` before intersecting with Canny edges, cutting background edges.
  - Contour vetting enforces `len(c) >= 5` (otherwise `cv2.fitEllipse` fails) and `area >= ellipse_min_contour_area` (parameter, default 150). The best ellipse maximizes area; previous-ellipse similarity checks are currently disabled (`self.prev_ellipse = None` placeholder) but show how additional gating could be added.
  - When `det_bbox` is absent (e.g., during `ELLIPSE_TRACK`), the entire mask is valid, so depth filtering drives the search region; fallback masks tighten it again by dilating around prior ellipses.
- **Servo use of centroids:**
  - Pitch/yaw errors are raw pixel offsets scaled by `5 * error / image_dim` when the offsets exceed half the image width/height (`x_max`, `y_max`). This avoids jitter from tiny motions while still responding when the ellipse or detection drifts significantly.
  - Commands publish only when the angle actually changes, minimizing redundant ROS traffic and actuator chatter.
