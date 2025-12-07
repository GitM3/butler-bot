import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file("/home/zander/Documents/20251202_164751.bag")

align = rs.align(rs.stream.color)

pipeline.start(config)
spatial    = rs.spatial_filter()
temporal   = rs.temporal_filter()
holefill   = rs.hole_filling_filter()
to_disp    = rs.disparity_transform(True)
to_depth   = rs.disparity_transform(False)

spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 10)
spatial.set_option(rs.option.holes_fill, 0)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

window = "Overlay"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.createTrackbar("Depth Thresh (mm)", window, 1200, 3000, lambda x: None)
cv2.createTrackbar("Margin (mm)", window, 20, 100, lambda x: None)
cv2.createTrackbar("Kernel", window, 10, 80, lambda x: None)
cv2.createTrackbar("Alpha1 (mask)", window, 40, 100, lambda x: None)
cv2.createTrackbar("Alpha2 (debug)", window, 40, 100, lambda x: None)

def draw_ellipse(img, ellipse, color=(0,255,0), thickness=2):
    if ellipse is None:
        return img
    output = img.copy()
    cv2.ellipse(output, ellipse, color, thickness)
    return output

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        frame = depth_frame
        frame = to_disp.process(frame)
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        frame = to_depth.process(frame)
        frame = holefill.process(frame)
        if not depth_frame or not color_frame:
            continue

        depth = np.asanyarray(frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        # Read parameters
        depth_thresh = cv2.getTrackbarPos("Depth Thresh (mm)", window)
        alpha1 = cv2.getTrackbarPos("Alpha1 (mask)", window) / 100.0
        alpha2 = cv2.getTrackbarPos("Alpha2 (debug)", window) / 100.0
        kernel_size = cv2.getTrackbarPos("Kernel", window)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))


        init_mask = ((depth > 0) & (depth <= depth_thresh)).astype(np.uint8) * 255
        mask = cv2.dilate(init_mask, kernel, iterations=1)
        close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.putText(color, "X", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        mask_color = cv2.cvtColor(close_mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(color, 1 - alpha1, mask_color, alpha1, 0)
        #overlay = cv2.addWeighted(overlay, 1 - alpha2, debug_color, alpha2, 0)

        combined = np.hstack((color, overlay))
        cv2.imshow(window, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
