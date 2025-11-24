from __future__ import print_function

import argparse

import cv2 as cv

max_lowThreshold = 100
max_highThreshold = 300
window_name = 'Edge Map'
ratio = 3
kernel_size = 3
alpha = 1
low_threshold = 10
high_threshold = 30

def find_bottom_ellipse(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_ellipse, best_area = None, 0
    for c in contours:
        if len(c) >= 5:
            area = cv.contourArea(c)
            if area > 50:  # ignore noise
                ellipse = cv.fitEllipse(c)
                if area > best_area:
                    best_area, best_ellipse = area, ellipse

    return best_ellipse


def setLow(val):
    global low_threshold
    low_threshold = val
    CannyThreshold()

def setHigh(val):
    global high_threshold
    high_threshold = val
    CannyThreshold()

def CannyThreshold():
    global high_threshold
    global low_threshold
    print(f"Low, high: {low_threshold},{high_threshold}")
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, high_threshold, kernel_size)
    ellipse = find_bottom_ellipse(detected_edges)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    if ellipse is not None:
        (cx, cy), (MA, ma), angle = ellipse
        cv.ellipse(dst, ellipse, (0,255,0), 2)
        cv.circle(dst, (int(cx), int(cy)), 3, (0,0,255), -1)
    cv.imshow(window_name, dst)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar("Min Threshold", window_name , 0, max_lowThreshold, setLow)
cv.createTrackbar("Max Threshold", window_name , 0, max_lowThreshold, setHigh)
CannyThreshold()
while True:

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
