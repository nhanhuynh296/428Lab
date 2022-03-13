# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--video_path", default="test_video_yonex.mp4", help="Process a video. Reads all standard formats.")
parser.add_argument("--image_delay", default=25, help="The image delay for rendering the next frame")
parser.add_argument("--buffer", default=64, help="The deque buffer for storing ball locations")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Main Program
light_white = (0,0,200)
# light_white = (0,0,0)
dark_white = (145,60,255)

video_source = args[0].video_path
image_delay = args[0].image_delay

cap = cv2.VideoCapture(video_source)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo_output.avi', fourcc, 25, (1280, 720))

# Lets try to load OpenPose first
try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
except Exception as e:
    print(e)
    sys.exit(-1)


def difference_with_centroid():
    # cap = cv2.VideoCapture(-1)  # Open the webcam device.
    cap = cv2.VideoCapture("test_video_yonex.mp4")  # Open the webcam device.

    # Load two initial images from the webcam to begin.
    ret, img0 = cap.read()
    ret, img1 = cap.read()

    while True:
        # Calculate the differences of the two images.
        diff = cv2.subtract(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        ret, diff = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

        # Move the data in img0 to img1. Uncomment this line for differencing from the first frame.
        img1 = img0 
        ret, img0 = cap.read()  # Grab a new frame from the camera for img0.

        datum = op.Datum()
        datum.cvInputData = img0 # Setup the datum object
        opWrapper.emplaceAndPop([datum]) # Run openpose to add annotated body parts

        # print("Body keypoints: \n" + str(datum.poseKeypoints))

        annotatedFrame = datum.cvOutputData # Obtain the annotated frame

        if not ret:
            break

        hsv = cv2.cvtColor(annotatedFrame, cv2.COLOR_BGR2HSV)

        # Use the moments of the difference image to draw the centroid of the difference image.
        moments = cv2.moments(diff)
        # mask = cv2.inRange(hsv, (0,0,150), (60,255,255))
        mask = cv2.inRange(hsv, light_white, dark_white)

        # Dilation to expand white blobs
        kernel1 = np.ones((3,3), np.uint8)
        dilation = cv2.morphologyEx(diff, cv2.MORPH_DILATE, kernel1)
        kernel2 = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
        kernel3 = np.ones((5,5), np.uint8)
        closure = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel3)

        rgb = cv2.cvtColor(img0, cv2.COLOR_HSV2BGR)
        # opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        # if moments["m00"] != 0:  # Check for divide by zero errors.
        #     cX = int(moments["m10"] / moments["m00"])
        #     cY = int(moments["m01"] / moments["m00"])
        #     # cv2.circle(diff, (cX, cY), 7, (255, 0, 0), -1)
        #     cv2.circle(img0, (cX, cY), 7, (255, 0, 0), -1)

        # cv2.imshow('Difference', diff)  # Display the difference to the screen.
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Dilation', dilation)
        # cv2.imshow('Opening', opening)
        # cv2.imshow('Closure', closure)
        cv2.imshow('Shuttle Tracking', rgb)
        res = cv2.bitwise_and(annotatedFrame, annotatedFrame, mask=closure)
        
        positions = cv2.findNonZero(closure)
        if positions is not None:
            for point in positions:
                # print(point)
                x,y = point[0]
                cv2.circle(annotatedFrame, (x,y), 2, (255, 0, 0), -1)
        # if positions and positions.all():
        #     continue
        # for position in positions:
        #     if position:
        #         x,y = positon
        #         cv2.circle(hsv, (x,y), 2, (255,0,0), -1)
        # print("----\n{}\n----".format(positions))

        cv2.imshow('Image', annotatedFrame)
        out.write(annotatedFrame)
        # cv2.imshow('Difference', img0)  # Display the difference to the screen.

        # Close the script when q is pressed.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

difference_with_centroid()