from __future__ import print_function

import argparse

import cv2 as cv

window_name = 'Ellipse Debug'

# ----------------------------
# Default parameter values
# ----------------------------
params = {
    "bottom_ratio":     25,   # slider 0–100 → /100
    "min_contour_area": 200,  # slider 0–5000
    "canny1":           50,
    "canny2":           150,
    "gauss_kernel":     5,    # odd enforced
    "max_axis_scale":   150,  # slider → /100
    "min_axis_length":  20,
    "axis_ratio_min":   30    # slider → /100
}

# --------------------------------------
# Ellipse detection using your parameters
# --------------------------------------
def find_bottom_ellipse(gray, p):
    h = gray.shape[0]
    roi_h = int(h * (p["bottom_ratio"] / 100.0))

    # region of interest is lower slice
    roi = gray[h - roi_h : h, :]

    # gaussian smoothing
    k = int(p["gauss_kernel"])
    if k % 2 == 0:
        k += 1
    blur = cv.GaussianBlur(roi, (k, k), 0)

    edges = cv.Canny(blur, p["canny1"], p["canny2"])

    # find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_ellipse, best_area = None, 0
    for c in contours:
        if len(c) < 5:
            continue

        area = cv.contourArea(c)
        if area < p["min_contour_area"]:
            continue

        ellipse = cv.fitEllipse(c)
        (_, _), (MA, ma), _ = ellipse

        # enforce axis limits
        if MA < p["min_axis_length"] or ma < p["min_axis_length"]:
            continue

        ratio = min(MA, ma) / max(MA, ma)
        if ratio < p["axis_ratio_min"] / 100.0:
            continue

        # Optional: reject extremely large ellipses
        # scale relative to ROI dimensions
        max_allowed = p["max_axis_scale"] / 100.0
        if MA > roi.shape[1] * max_allowed or ma > roi.shape[0] * max_allowed:
            continue

        if area > best_area:
            best_area, best_ellipse = area, ellipse

    return best_ellipse, edges, roi_h


# --------------------------------------
# Recompute + refresh display
# --------------------------------------
def update(_=None):
    p = params

    ellipse, edges, roi_h = find_bottom_ellipse(src_gray, p)

    # Convert edges to 3-channel so we can draw colors on it
    display = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    h, w = src_gray.shape[:2]

    # visualize ROI (blue box)
    cv.rectangle(display,
                 (0, h - roi_h),
                 (w, h),
                 (255, 0, 0), 1)

    # draw ellipse and center (green + red)
    if ellipse is not None:
        (cx, cy), (MA, ma), angle = ellipse

        # ellipse was fit in ROI coordinates → adjust Y
        cy += (h - roi_h)
        adj_ellipse = ((cx, cy), (MA, ma), angle)

        cv.ellipse(display, adj_ellipse, (0, 255, 0), 2)
        cv.circle(display, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    cv.imshow(window_name, display)


# --------------------------------------
# Slider callbacks → update params
# --------------------------------------
def set_bottom_ratio(v):
    params["bottom_ratio"] = v
    update()

def set_min_area(v):
    params["min_contour_area"] = max(0, v)
    update()

def set_canny1(v):
    params["canny1"] = v
    update()

def set_canny2(v):
    params["canny2"] = v
    update()

def set_gauss(v):
    # ensure odd kernel
    v = max(1, v)
    if v % 2 == 0:
        v += 1
    params["gauss_kernel"] = v
    update()

def set_axis_scale(v):
    params["max_axis_scale"] = v
    update()

def set_min_axis_len(v):
    params["min_axis_length"] = v
    update()

def set_axis_ratio(v):
    params["axis_ratio_min"] = v
    update()


# --------------------------------------
# Startup
# --------------------------------------
parser = argparse.ArgumentParser(description='Ellipse parameter tester.')
parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

cv.namedWindow(window_name)

# --- Create sliders ---
cv.createTrackbar("bottom_ratio %", window_name, params["bottom_ratio"], 100, set_bottom_ratio)
cv.createTrackbar("min_contour_area", window_name, params["min_contour_area"], 5000, set_min_area)
cv.createTrackbar("canny1", window_name, params["canny1"], 300, set_canny1)
cv.createTrackbar("canny2", window_name, params["canny2"], 500, set_canny2)
cv.createTrackbar("gaussian_kernel", window_name, params["gauss_kernel"], 31, set_gauss)
cv.createTrackbar("max_axis_scale %", window_name, params["max_axis_scale"], 300, set_axis_scale)
cv.createTrackbar("min_axis_length px", window_name, params["min_axis_length"], 500, set_min_axis_len)
cv.createTrackbar("axis_ratio_min %", window_name, params["axis_ratio_min"], 100, set_axis_ratio)

update()

# --- Quit on 'q' ---
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
