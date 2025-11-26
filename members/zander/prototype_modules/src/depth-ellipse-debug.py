import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align = rs.align(rs.stream.color)

pipeline.start(config)

window = "Overlay"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)

cv2.createTrackbar("Depth Thresh", window, 1000, 5000, lambda x: None)
cv2.createTrackbar("Alpha", window, 50, 100, lambda x: None)
cv2.createTrackbar("kernel", window, 1, 50, lambda x: None)
cv2.createTrackbar("c1", window, 1, 150, lambda x: None)
cv2.createTrackbar("c2", window, 1, 150, lambda x: None)

def find_ellipse_from_masked_rgb(mask, rgb):
    masked = cv2.bitwise_and(rgb, rgb, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    c1  = cv2.getTrackbarPos("c1", window)
    c2  = cv2.getTrackbarPos("c2", window)
    edges = cv2.Canny(gray_blur, c1, c2)

    edges = cv2.bitwise_and(edges, edges, mask=mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse = None
    best_area = 0

    for c in contours:
        if len(c) < 5:
            continue

        area = cv2.contourArea(c)
        if area < 200:     # reject small/noisy
            continue

        ellipse = cv2.fitEllipse(c)

        if area > best_area:
            best_area = area
            best_ellipse = ellipse

    return best_ellipse

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

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())      # uint16 depth in mm
        color_image = np.asanyarray(color_frame.get_data())      # RGB aligned to depth

        depth_thresh_mm = cv2.getTrackbarPos("Depth Thresh", window)
        alpha_percent   = cv2.getTrackbarPos("Alpha", window)
        kernel_size  = cv2.getTrackbarPos("kernel", window)
        alpha = alpha_percent / 100.0

        init_mask = ((depth_image > 0) & (depth_image <= depth_thresh_mm)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        mask = cv2.dilate(init_mask, kernel, iterations=1)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ellipse = find_ellipse_from_masked_rgb(mask, color_image)

        vis = draw_ellipse(color_image, ellipse)
        cv2.imshow("Overlay", vis)
        overlay = cv2.addWeighted(color_image, 1 - alpha, mask_3ch, alpha, 0)
        images = np.hstack((vis, overlay))
        cv2.imshow(window, images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
