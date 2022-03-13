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

video_source = args[0].video_path
image_delay = args[0].image_delay

cap = cv2.VideoCapture(video_source)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 25, (1280, 720))

# Lets try to load OpenPose first
try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
except Exception as e:
    print(e)
    sys.exit(-1)

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
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame # Setup the datum object
    opWrapper.emplaceAndPop([datum]) # Run openpose to add annotated body parts

    # print("Body keypoints: \n" + str(datum.poseKeypoints))

    annotatedFrame = datum.cvOutputData # Obtain the annotated frame
    
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
    cv2.imshow("Badminton Training", annotatedFrame)
    if cv2.waitKey(image_delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

