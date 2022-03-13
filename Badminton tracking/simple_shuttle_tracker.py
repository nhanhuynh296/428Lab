# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import copy
import itertools

# Flags
parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--video_path", default="test_video_yonex.mp4", help="Process a video. Reads all standard formats.")
parser.add_argument("--image_delay", default=1, help="The image delay for rendering the next frame")
parser.add_argument("--buffer", default=64, help="The deque buffer for storing ball locations")
args = parser.parse_known_args()

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Main Program

video_source = args[0].video_path
image_delay = args[0].image_delay

cap = cv2.VideoCapture(video_source)

# nlog(n) way to get "close" items from iterations
def closest_array_items(a1, a2, min_dif):
    
    if len(a1) == 0 or len(a2) == 0:
        print("Array empty")
        return []
    print("Array not empty")
    print([x.pt for x in a2])
    a1, a2  = iter(sorted(a1)), iter(sorted([x.pt for x in a2]))
    i1, i2 = a1.next(), a2.next()
    pairs = []
    # min_dif = float('inf')
    while True:
        dif = abs(i1 - i2)
        print(dif)
        if dif < min_dif:
            #  min_dif = dif
             pair = i1, i2
             pairs.append(pair)
             if not min_dif:
                  break
        if i1 > i2:
            try:
                i2 = a2.next()
            except StopIteration:
                break
        else:
            try:
                i1 = a1.next()
            except StopIteration:
                break
    return pairs

def find_shuttle(img, lower, upper):
    mask = cv2.inRange(img, lower, upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=len)
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = np.array([int(x), int(y)])
        radius = int(radius)
        return center, radius
    else:
        return None, None

def onChangeFunction(value): {
    print("Trackbar changed to: {}".format(value))
}

# timestep = 1/30 # Default framerate is 30fps
# lower = np.array([230,230,230], dtype="uint8")
# upper = np.array([255,255,255], dtype="uint8")

# kalman = cv2.KalmanFilter(4,2)
# kalman.transitionMatrix = np.array([[1, 0, timestep, 0], [0, 1, 0, timestep], [0,0,1,0], [0,0,0,1]], np.float32)
# kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
# kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,100,0],[0,0,0,100]], np.float32)
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32)
# kalman.statePost = np.array([0,0,0,0], np.float32)
# kalman.errorCovPost = np.array([[1000,0,0,0],[0,1000,0,0],[0,0,1000,0],[0,0,0,1000]])

# Actual Processing of image
cv2.namedWindow('Canny Edge Detection')
cv2.namedWindow('Blob detection')
cv2.namedWindow('Color Thresholding + Morphology')

cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 183, 1200, onChangeFunction)
cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 275, 1200, onChangeFunction)
cv2.createTrackbar('noiseReduction', 'Canny Edge Detection', 10, 15, onChangeFunction)

cv2.createTrackbar('MinRadius', 'Blob detection', 0, 100, onChangeFunction)
cv2.createTrackbar('MaxRadius', 'Blob detection', 41, 100, onChangeFunction)

# cv2.createTrackbar('lowerColorH', 'Color Thresholding + Morphology', 0, 255, onChangeFunction)
# cv2.createTrackbar('lowerColorS', 'Color Thresholding + Morphology', 0, 255, onChangeFunction)
# cv2.createTrackbar('lowerColorV', 'Color Thresholding + Morphology', 200, 255, onChangeFunction)

# cv2.createTrackbar('upperColorH', 'Color Thresholding + Morphology', 145, 255, onChangeFunction)
# cv2.createTrackbar('upperColorS', 'Color Thresholding + Morphology', 60, 255, onChangeFunction)
# cv2.createTrackbar('upperColorV', 'Color Thresholding + Morphology', 255, 255, onChangeFunction)

cv2.createTrackbar('lowerColorH', 'Color Thresholding + Morphology', 0, 255, onChangeFunction)
cv2.createTrackbar('lowerColorS', 'Color Thresholding + Morphology', 0, 255, onChangeFunction)
cv2.createTrackbar('lowerColorV', 'Color Thresholding + Morphology', 143, 255, onChangeFunction)

cv2.createTrackbar('upperColorH', 'Color Thresholding + Morphology', 104, 255, onChangeFunction)
cv2.createTrackbar('upperColorS', 'Color Thresholding + Morphology', 218, 255, onChangeFunction)
cv2.createTrackbar('upperColorV', 'Color Thresholding + Morphology', 172, 255, onChangeFunction)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        cap = cv2.VideoCapture(video_source)
        continue
    
    # Canny Edge
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')
    
    edges = cv2.Canny(frame, threshold1, threshold2)
    cv2.imshow('Canny Edge Detection', edges)

    nextRet, nextFrame = cap.read()
    if not nextRet:
        cap = cv2.VideoCapture(video_source)
        continue
    
    # Image differencing
    nextEdges = cv2.Canny(nextFrame, threshold1, threshold2)
    diff = cv2.subtract(edges, nextEdges)
    h = cv2.getTrackbarPos('noiseReduction', 'Canny Edge Detection')
    #     # Dilation to expand white blobs
    #     kernel1 = np.ones((3,3), np.uint8)
    #     dilation = cv2.morphologyEx(diff, cv2.MORPH_DILATE, kernel1)
    #     kernel2 = np.ones((5,5), np.uint8)
    #     opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
    #     kernel3 = np.ones((5,5), np.uint8)
    #     diff = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel3)
    diff = cv2.fastNlMeansDenoising(diff, None, h=h)

    cv2.imshow('Canny + Difference', diff)

    # Blob detection
    minRadius = cv2.getTrackbarPos('MinRadius', 'Blob detection') / 100
    maxRadius = cv2.getTrackbarPos('MaxRadius', 'Blob detection') / 100
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByColor = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minArea = 3.14159 * minRadius * minRadius
    params.maxArea = 3.14159 * maxRadius * maxRadius

    blobDetector = cv2.SimpleBlobDetector_create(params)
    keypoints = blobDetector.detect(diff)
    frame_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Blob detection', frame_keypoints)

    # Color detection
    lowerColorH = cv2.getTrackbarPos('lowerColorH', 'Color Thresholding + Morphology')
    lowerColorS = cv2.getTrackbarPos('lowerColorS', 'Color Thresholding + Morphology')
    lowerColorV = cv2.getTrackbarPos('lowerColorV', 'Color Thresholding + Morphology')
    
    upperColorH = cv2.getTrackbarPos('upperColorH', 'Color Thresholding + Morphology')
    upperColorS = cv2.getTrackbarPos('upperColorS', 'Color Thresholding + Morphology')
    upperColorV = cv2.getTrackbarPos('upperColorV', 'Color Thresholding + Morphology')
    
    lowerBoundColor = (lowerColorH, lowerColorS, lowerColorV)
    upperBoundColor = (upperColorH, upperColorS, upperColorV)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBoundColor, upperBoundColor)


    kernel = np.ones((3,3), np.uint8)
    # colorFrame = cv2.bitwise_and(hsv, hsv, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = copy.deepcopy(frame)
    for contour in contours:
        # contour = max(contours, key=len)
        (x,y), radius = cv2.minEnclosingCircle(contour)
        center = np.array([int(x), int(y)])
        if center is not None:
            cv2.circle(frameCopy, tuple(center), int(radius), (0, 255, 0), 2) # Circle around object
            # cv2.circle(frameCopy, tuple(center), 1, (0, 255, 0), 2) # Center circle
    cv2.imshow('Color Thresholding + Morphology', frameCopy)

    # Combine Techniques

    # TODO: Uncomment out the morphology stuff from below

    result = cv2.KeyPoint_convert(keypoints)
    result_points = cv2.bitwise_and(result, result, mask) # Just applying mask to the image
    combineFrameCopy = copy.deepcopy(frame)

    if result is not None:
        for result in result_points:
            cv2.circle(combineFrameCopy, tuple(result), 1, (0, 255, 0), 2)

    cv2.imshow('Combined Image', combineFrameCopy)

    # print(contours)

    # results = closest_array_items(contours, keypoints, 20)
    # for point in results:
    #     print(point)

    # cv2.imshow('Color Thresholding + Morphology', colorFrame)
    #  hsv = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2HSV)

    #     # Use the moments of the difference image to draw the centroid of the difference image.
    #     moments = cv2.moments(diff)
    #     # mask = cv2.inRange(hsv, (0,0,150), (60,255,255))
    #     mask = cv2.inRange(hsv, light_white, dark_white)

    #     # Dilation to expand white blobs
    #     kernel1 = np.ones((3,3), np.uint8)
    #     dilation = cv2.morphologyEx(diff, cv2.MORPH_DILATE, kernel1)
    #     kernel2 = np.ones((5,5), np.uint8)
    #     opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
    #     kernel3 = np.ones((5,5), np.uint8)
    #     closure = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel3)

    #     rgb = cv2.cvtColor(img0, cv2.COLOR_HSV2BGR)
    #     # opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    #     # if moments["m00"] != 0:  # Check for divide by zero errors.
    #     #     cX = int(moments["m10"] / moments["m00"])
    #     #     cY = int(moments["m01"] / moments["m00"])
    #     #     # cv2.circle(diff, (cX, cY), 7, (255, 0, 0), -1)
    #     #     cv2.circle(img0, (cX, cY), 7, (255, 0, 0), -1)

    #     # cv2.imshow('Difference', diff)  # Display the difference to the screen.
    #     # cv2.imshow('Mask', mask)
    #     # cv2.imshow('Dilation', dilation)
    #     # cv2.imshow('Opening', opening)
    #     # cv2.imshow('Closure', closure)
    #     cv2.imshow('Shuttle Tracking', rgb)
    #     res = cv2.bitwise_and(annotatedFrame, annotatedFrame, mask=closure)
        
    #     positions = cv2.findNonZero(closure)
    #     if positions is not None:
    #         for point in positions:
    #             # print(point)
    #             x,y = point[0]
    #             cv2.circle(annotatedFrame, (x,y), 2, (255, 0, 0), -1)

    if cv2.waitKey(image_delay) & 0xFF == ord('q'):
        break
    
    
    # Find shuttle in frame
    # center, radius = find_shuttle(frame, lower, upper)
    # print('Mesaurement:', center)
    # predicted_state = kalman.predict()
    # center_ = (predicted_state[0], predicted_state[1])
    
    # axis_lengths = (kalman.errorCovPre[0,0], kalman.errorCovPre[1,1])
    # cv2.ellipse(annotatedFrame, center_, axis_lengths, 0, 0, 360, color=(255,0,0))

    # if center is not None and radius is not None:
    #     cv2.circle(annotatedFrame, tuple(center), radius, (0,255,0), 2)
    #     cv2.circle(annotatedFrame, tuple(center), 1, (0,255,0), 2)

    #     measured = np.array([[center[0]], center[1]], dtype="float32")
    #     estimated_state = kalman.correct(mesaured)

    # out.write(annotatedFrame)
    # cv2.imshow("Badminton Training", annotatedFrame)
    # if cv2.waitKey(image_delay) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

