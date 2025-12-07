import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file("/home/zander/Documents/20251202_164751.bag")
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align = rs.align(rs.stream.color)

pipeline.start(config)

window = "Overlay"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)

cv2.createTrackbar("Depth Thresh", window, 1000, 5000, lambda x: None)
cv2.createTrackbar("Alpha1", window, 50, 100, lambda x: None)
cv2.createTrackbar("Alpha2", window, 50, 100, lambda x: None)
cv2.createTrackbar("kernel", window, 50, 100, lambda x: None)
cv2.createTrackbar("c1", window, 10, 150, lambda x: None)
cv2.createTrackbar("c2", window, 50, 150, lambda x: None)

def sanitize_ellipse(ellipse):
    if ellipse is None:
        return None
    (cx, cy), (ma, mi), angle = ellipse
    cx, cy, ma, mi, angle = map(float, [cx, cy, ma, mi, angle])

    if np.isnan(cx) or np.isnan(cy) or np.isnan(ma) or np.isnan(mi) or np.isnan(angle):
        return None
    if ma <= 1 or mi <= 1:
        return None
    if ma > 2000 or mi > 2000:  # sanity bound
        return None

    angle = angle % 180.0

    if mi > ma:
        ma, mi = mi, ma

    return ((cx, cy), (ma, mi), angle)


def find_ellipse_from_masked_rgb(mask, rgb):
    masked = cv2.bitwise_and(rgb, rgb, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    c1  = cv2.getTrackbarPos("c1", window)
    c2  = cv2.getTrackbarPos("c2", window)
    edges = cv2.Canny(gray_blur, c1, c2)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    canny_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best_ellipse = None
    best_area = 0

    for c in contours:
        if len(c) < 5:
            continue

        area = cv2.contourArea(c)
        if area < 200:     # reject small/noisy
            continue

        ellipse = cv2.fitEllipseAMS(c)
        if area > best_area:
            best_area = area
            best_ellipse = ellipse

    return best_ellipse, canny_frame

def draw_ellipse(img, ellipse, color=(0,255,0), thickness=2):
    if ellipse is None:
        return img
    output = img.copy()
    ellipse = sanitize_ellipse(ellipse)
    cv2.ellipse(output, ellipse, color, thickness)
    return output

# Constant velocity
# x(t+1) = x(t) + v(t) * dt
# v(t+1) = v(t)

kf = cv2.KalmanFilter(5*2,5) # cx,cy,ma,mi,an
dt = 1.0/30.0
mat_i  = np.eye(5, dtype=np.float32)
mat_dt = mat_i * dt

A = np.zeros((10, 10), dtype=np.float32)
A[0:5, 0:5] = mat_i
A[0:5, 5:10] = mat_dt
A[5:10, 5:10] = mat_i
kf.transitionMatrix = A

H = np.zeros((5,10), dtype=np.float32)
H[0:5,0:5] = mat_i
kf.measurementMatrix = H

kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-2
kf.measurementNoiseCov = np.eye(5, dtype=np.float32) * 1e-1
kf.errorCovPost = np.eye(10, dtype=np.float32) * 1

def kf_step(ellipse):
    pred = kf.predict()
    px, py          = float(pred[0]), float(pred[1])
    p_major         = float(pred[2])
    p_minor         = float(pred[3])
    p_angle_deg     = float(pred[4])

    tracked_ellipse = ((px, py), (p_major, p_minor), p_angle_deg)

    if ellipse is not None:
        (cx, cy), (major, minor), angle_deg = ellipse
        meas = np.array(
            [[cx], [cy], [major], [minor], [angle_deg]],
            dtype=np.float32
        )
        kf.correct(meas)
        # tracked_ellipse = ((float(cx), float(cy)),
        #                    (float(major), float(minor)),
        #                    float(angle_deg))

    return tracked_ellipse

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
        alpha1_percent   = cv2.getTrackbarPos("Alpha1", window)
        alpha2_percent   = cv2.getTrackbarPos("Alpha2", window)
        kernel_size  = cv2.getTrackbarPos("kernel", window)
        alpha1 = alpha1_percent / 100.0
        alpha2 = alpha2_percent / 100.0

        init_mask = ((depth_image > 0) & (depth_image <= depth_thresh_mm)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        mask = cv2.dilate(init_mask, kernel, iterations=1)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ellipse,canny_frame = find_ellipse_from_masked_rgb(mask, color_image)

        tracked_ellipse = kf_step(ellipse)

        vis = draw_ellipse(color_image, ellipse,color=(255,0,0))
        vis = draw_ellipse(vis, tracked_ellipse,color=(0,0,255))

        overlay = cv2.addWeighted(color_image, 1 - alpha1, mask_3ch, alpha1, 0)
        overlay = cv2.addWeighted(overlay, 1 - alpha2, canny_frame, alpha2, 0)
        images = np.hstack((vis, overlay))
        cv2.imshow(window, images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
