import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align = rs.align(rs.stream.color)

pipeline.start(config)

window_mask = "Depth Mask"
window_overlay = "Overlay"

cv2.namedWindow(window_mask, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_overlay, cv2.WINDOW_NORMAL)

cv2.createTrackbar("Depth Thresh", window_mask, 1000, 5000, lambda x: None)
cv2.createTrackbar("Alpha", window_mask, 50, 100, lambda x: None)


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())      # uint16 depth in mm
        color_image = np.asanyarray(color_frame.get_data())      # RGB aligned to depth

        depth_thresh_mm = cv2.getTrackbarPos("Depth Thresh", window_mask)
        alpha_percent   = cv2.getTrackbarPos("Alpha", window_mask)
        alpha = alpha_percent / 100.0

        mask = ((depth_image > 0) & (depth_image <= depth_thresh_mm)).astype(np.uint8) * 255
        cv2.imshow(window_mask, mask)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(color_image, 1 - alpha, mask_3ch, alpha, 0)
        cv2.imshow(window_overlay, overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
