# COSC428 Lab 1 - Working with Cameras and Basic Image Processing

## Objectives
The overall goal of this lab is to introduce **OpenCV code for displaying, recording, and replaying video, and how to execute basic image processing functions**.

Given this background, you will then use **tools to change camera attributes and edit video** and you will **calibrate a camera to remove radial distortion** (fisheye lens distortion).

Finally, you will learn some basic OpenCV image processing functions such as finding:
- edges
- lines
- circles
- faces
- centroids
- difference-images
- fiducial markers. 
You will also sharpen images and reduce noise using morphology.  

You will need to read and run some small applications written in Python. During this lab, you should enhance your understanding some concepts of computer vision video processing by implementing Python programs that demonstrate these concepts.

## Preparation
Do not install OpenCV on Lab 3 PCs because OpenCV is already installed on these PCs.

Download the "Lab 1 files" zip archive from Learn, or from the [Git repository on eng-git](https://eng-git.canterbury.ac.nz/owb14/cosc428-lab1). The following files should be in that zip file:
- images/
- camera_driver.py
- canny.py
- checkerboard.png
- differencing.py
- face_detection.py
- fiducial.png
- fiducial.py
- fiducial2.png
- find_camera_calibration.py
- haarcascade_eye.xml
- haarcascade_frontalface_default.xml
- hough_circle.py
- hough_line.py
- lab01_handout.md
- load_camera_calibration.py
- load_video_file.py
- morphology.py
- segmentation.py
- video_from_webcam.py
- webcam_to_file.py


## Troubleshooting
- If you get a video I/O error, change the “0” in VideoCapture to “-1” which grabs the first available camera.
- If you still cannot access the camera at any time, try unplugging it and plugging it back in.
- You can check your camera exists on Linux with: `ls /dev/video*` in the bash terminal.
- If you get error `QObject::moveToThread: Current thread is not the object's thread. Cannot move to target thread` on a Lab 3 PC, then you probably accidentally installed OpenCV. (To fix, pip3 uninstall opencv-python (or pip uninstall opencv-python))

## Display Video from a Webcam
One of the most basic tasks in OpenCV is to simply start retrieving video data from the camera so that we can start providing our algorithms with some data. At the moment, however, we don’t have any data, so we’ll start withshowing the video to the screen for now.

The integer passed into the cv2.VideoCapture() function specifies the camera used. For a computer with a single camera, using 0 will work. Otherwise, you might need to do some testing to determine which device it is that you wish to use.

### To do:
Run `video_from_webcam.py`

Add the following line of code just before imshow() and observe the effect:
```
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```


## Save Video from a Webcam to a File
It is convenient to save video from a webcam to file for use as a consistent test file while tweaking an algorithm. This code stores the video in an .avi file using the XVID codec, but the commands for other containers and codecs can be found in the OpenCV documentation.

### To do:
Run `webcam_to_file.py` and check that an `output.avi` file was created in your local directory.

Play `output.avi` to check that it recorded correctly.


## Load a video from a File
Now that we have stored our video, we need a way of opening it again. As it happens, the method for capturing the video from a webcam is the same for opening a video file. Instead of specifying a number for the camera, we pass in a file name instead. 

In this example, the frame is converted to grayscale before being displayed, but in practice that is where you would perform your algorithm.

### To do:
Run ` load_video_file.py`
Modify the code to skip the `cv2.cvtColor` line and observe the difference.


## Camera Calibration
Perfection is hard, and isn’t typically cheap either. As a result, the lens and image sensor of a webcam differ slightly from device to device. Camera calibration allows one to correct distortions caused by these irregularities. The calibration parameters generated by this process can then be saved and loaded when the camera is used for a computer vision algorithm.

While modern cameras are relatively consistent, and can seem to be free from most distortion effects, even subtle changes can cause problems. For example, if an algorithm uses straight edges to detect a given feature, if there is a radial distortion introduced into the image as a result of a lens imperfection, then the algorithm may not perform correctly.

To calibrate the camera, run the `find_camera_calibration.py` script, and hold a checkerboard image in front of the camera and **ensure that the *entire* checkerboard is visible at all times**. To get a good calibration result, it is important to rotate the checkerboard, and put it at angles to the webcam, while also making sure that a range of poses are recorded all over the webcam’s field of view. (Tip: Lie the checkerboard flat on desk and move webcam.)

Once calibrated, the lines of the checkerboard should be completely straight when viewed through the webcam. If this is not the case, you should run the calibration again, taking care to ensure the checkerboard is flat during calibration.

A 6x9 checkerboard can be downloaded [here](https://docs.opencv.org/2.4/_downloads/pattern.png). Note that it isn't necessary to print the checkerboard. It works just fine if shown on a monitor instead.

### Generating the Calibration Files
Here the camera calibration is performed with the checkerboard pattern. There is a short delay after every frame where the checkerboard is detected to limit the number of frames we need to process later.

The two calibration files “camera_matrix.npy” and “distortion_coeff.npy” are saved for later use.

#### To do:
Run `find_camera_calibration.py` and collect about six images at completely different angles and positions in the image, and **then hold down “q” until “Calculating Camera Distortion...” is displayed.**

(Note: If you have changed brightness, contrast or saturation, change them all back to default or 50%.) 

You can compare your values in your two calibration files “camera_matrix.npy” and “distortion_coeff.npy” with those generated for one of the cameras in the labs:
```
camera_matrix = np.array([[ 1.10203699+03,  0.00000000e+00,  2.97856040+02,
                            0.00000000+00,  1.10715227e+03,  219618658e+02,
                            0.00000000+00,  0.000000000+00,  1.00000000+00,   
]])
distortion_coeff = np.array([7.978574896538845329e+02, 3.400042995004967317e+00, 
                            -1.786514214937548820e-02, -3.217060871280347668e-03,
                            -2.063856972981825777e+01])
```

Note that the point of this exercise is to demonstrate how to correct for camera distortions. It is not necessary to understand the details of every function and variable within `find_camera_calibration.py`.

### Loading the Camera Calibration
This shows how to load the calibration data for a camera and using it to produce an image that has been corrected for defects in the camera.

#### To do:
Run `load_camera_calibration.py`and verify that the lines in the displayed image are straight.
Try replacing the `dst` parameter in `imshow()` with `frame` and look for the distortion difference in straight lines in the resulting images. It can help to look near the edge of the camera with and without the correction.


## Canny Edge Detection
The Canny edge detector was developed in 1986 by John F. Canny. It is a common first step in a range of more complex algorithms such as the Hough Line detector which follows. The Canny edge detection algorithm is a good edge filter because it responds to an edge, not noise in the image. When one of the thresholds (in the code below) is large, Canny detects large scale edges (and better noise suppression) and when this threshold is small, Canny detects fine features.

The characteristics that a good edge detector should have:
- Good detection: Filter responds to edges, not noise
- Good localization: Detect edge near a true edge
- Single response: One detection per edge

Before running the Canny edge detector, first a **Gaussian filter** is used on the image, blurring some of the details. This is becasuse Canny is **susceptible to noise** present in a raw unprocessed image data.

An edge in an image may point in a variety of directions, so the Canny algorithm uses **four filters** to detect horizontal, vertical and diagonal edges in the blurred image. The edge detection operator (**Roberts, Prewitt, Sobel** for example) returns a value for the first derivative in the horizontal direction and the vertical direction. From this the edge gradient and direction can be determined. The edge direction angle is rounded to one of four angles representing vertical, horizontal and the two diagonals (0, 45, 90 and 135 degrees).

Steps of the Canny edge detection algorithm (using gradient directions found above):
1. **Norm of the gradient** (i.e. along direction of line/curve)
2. **Thresholding so as to respond to edges, _not noise_**
3. **Thinning for good localisation** and **only one response per edge** (for thinning, use non-maximum suppression, check if pixel is the **local maximum** along the gradient direction), **predict the next edge point** (assume the marked point is an edge point, then **construct the tangent to the edge curve** (which is normal to the gradient at that point) and use this to predict the next points)
4. **Hysteresis to improve _localisation_** (for hysteresis, check that maximum value of gradient value is sufficiently large. It is also possible to use a variable hysteresis **(start with a high threshold to begin edge curves and a low threshold to continue them**))

Thresholds:
- Gaussian kernel size, σ
   - Large σ detects larger scale edges (and suffers from less noise)
   - Small σ detects fine features
- Hystersis: The threshold used to determine if an edge is detected.

Further details can be found on the [OpenCV website](https://docs.opencv.org/4.5.0/da/d5c/tutorial_canny_detector.html) or on [Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector).

OpenCV has this edge detector as a single function. The example code shows a simple GUI with sliders to adjust the two thresholds. 

### To do:
Run `canny.py`, and play around with the threshold values.

It is recommended to use a roughly 2:1 or 3:1 ratio between the two thresholds. (Note that the raw values are important too, not just the ratio!)


## Hough Line Transform
Where the Canny edge detector looks for any edge in an image, the Hough (pronounced “huff”) Line Transform detects **straight** lines in an image. The Hough transform (HT) detects a line using a “voting” scheme where points vote for a set of parameters describing a line. The more votes for a particular set, the more evidence that the corresponding line is present in the image. So it can detect **multiple** lines in one shot.

The Hough Line Transform uses the "Hough space" to detect lines. This Hough space has two parameters:
- `d` - The shortest distance from the origin (generally (0,0) in the image) to a point on the line
- `θ` - The angle of the line.

Basic Hough transform algorithm
1. Initialize the Hough space, setting all of the array of all d,θ pairs to zero.
2. For each edge point I[x,y] in the image, find the angle and calculate the shortest distance to the origin. For that pair, increment the value of that pair in the Hough space by 1
    ```
    for θ = 0 to 180 
    d = x cosθ + y sinθ
    H[d, θ] += 1
    ```
3. Find the value(s) of (d, θ) where H[d, θ] is above the threshold
4. The detected lines in the image are given by d = x cosθ + y sinθ


OpenCV has two different functions for the Hough line transform:
- cv2.HoughLines (this is the normal Hough line detection algorithm)
- cv2.HoughLinesP (this is the probabilistic Hough line detection algorithm)

The HoughLinesP function is more efficient, and returns an array of start and end points for each line, whereas HoughLines simply returns Hesse normal form of the line (and thus does not limit the length of the detected line, nor give an idea of the length of the detected line segment). Also, the HoughLinesP function typically requires a lower threshold value.

Both functions are in the `hough_line.py` script in this lab. Note that while Canny edge detection was used as an initial pass, any image thresholded binarisation method will suffice.

The parameters for these functions are as follows:
- image – The binary image that is being processed.
- rho – The resolution of the r parameter for a line.
- theta – The resolution of the theta parameter for a line.
- threshold – The minimum number of votes that a line must get to be accepted.

The probabilistic Hough line detection algorithm also has two more parameters:
- minLineLength – The minimum length of a line for it to be accepted – which can be very useful for thresholding out tiny lines in an image.
- maxLineGap – The maximum gap between two points that can exist for them to be considered part of the same line.

For more details, see the [OpenCV documentation](https://docs.opencv.org/4.5.0/d2/d15/group__cudaimgproc__hough.html) or [Wikipedia](https://en.wikipedia.org/wiki/Hough_transform).

### To do:
- Run `hough_line.py`
- Swap between the two Hough Line algorithms by alternating the commented out lines and the end of the script.
- Observe the effects of playing around with the thresholds in the algorithms.
- Also look at the parameters specific to the probabalistic algorithm.
- Try loading in other images from the images/ directory and see how the detection process goes.


## Hough Circle Transform
The Hough cicle transform uses a similar method of voting in a special parameter space to detect circles instead of straight lines.

The Hough circle transform has the following parameters:
- image: The binary image being processed.
- method: The detection method (only cv2.HOUGH_GRADIENT is available).
- dp: The inverse ratio of the accumulator compared to the image resolution.
- minDist: The minimum distance between the centre of two detected circles. A too small value - may cause several the same feature in an image to be detected as several circles.
- param1: The first parameter passed to the Canny edge detector function.
- param2: The second parameter passed to the Canny edge detector function.
- minRadius: The minimum circle radius to be accepted.
- maxRadius: The maximum circle radius to be accepted.

### To do:
- Run `hough_circle.py`
- Vary the min and max radius to see how it influences the detection of circles in the image.
- Also try loading in other images from the images/ directory.


## Face Detection
OpenCV has a range of pre-trained classifiers for detecting faces. The code repository also contains the relevant classifiers (in .xml files) for this example, but there are more available in opencv/data/haarcascades, where the “opencv” directory is the repository OpenCV repository available from GitHub [here](https://github.com/opencv/opencv).

Additional details can be found on [Wikipedia](https://en.wikipedia.org/wiki/Viola–Jones_object_detection_framework).

### To do:  				
- Run `face_detection.py` 
- Rotate your head sideways and forward/backwards to see the limitations of this algorithm.


## Morphology
This example has four different functions representing four of the possible morphological operations available in OpenCV (erosion, dilation, opening and closing). When run, a GUI gives control of the Gaussian blur kernel size, and the number of iterations applied to the image. The resulting image is then shown in the GUI window.

Once again, additional details are available from [Wikipedia](https://en.wikipedia.org/wiki/Mathematical_morphology) and the [OpenCV documentation](https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html).

### To do:
Run the four functions in `morphology.py` and match them up to the following four sets of effects: [smoothes, removes spurs, and breaks narrow lines], [fills gaps and holes], [shrinks objects], [expands objects]. 


## Segmentation of a Colour Blob and Centroid Calculation
Image Segmentation is very useful for distinguishing features in an image. In the example code, several coins are segmented from the background and have the centroid of the segmented pixels drawn over the image. 

It can be seen from the “After Adaptive Threshold” image output that the threshold functions works well as a first step, but there is still a lot of noise in the output. Morphological functions are used to eliminate the noise around the outside of the coins, while preserving the general shape of the coins.

The centroid of the segmented contour is the mean location of all the points in the contour. The image moment of the contour can also be used to calculate the centroid as shown on lines 34-36. 
```
moments = cv2.moments(contour)
cX = int(moments["m10"] / moments["m00"])
cY = int(moments["m01"] / moments["m00"])
```

Also, it can be seen in the final result that since the contour fitting for the two coins in the top left is not very good, the calculated centroids for these blobs are also inaccurate.

### To do:
- Run `segmentation.py`
- Uncomment the cv2.drawContours function on line 29 and observe the contour where an ellipse is being fitted (drawn in red). While most of the coins have contours that match the whole coin well, the two in the top left are very rough and have a lot of smaller contours inside, rather than one large shape. 
- Attempt to tweak some of the morphological transformation parameters to fix this error and correctly detect all the coins (don't worry if you can't, the parameters have had a fair bit of optimisation at this stage).



## Frame Difference Images
Calculating the difference image for two frames is useful for highlighting what has changed between the two frames, or for detecting motion. Depending on the application, it might be useful to calculate the difference between two sequential frames, or by the difference between the first frame and the most recent frame. Commenting out line 17 in the difference() function will toggle between these two options: `img1 = img0`

When segmenting a moving object from a static background using a difference algorithm, there are some important terms to keep in mind:
- "Background subtraction" usually refers to the first frame, or some derivative of it, being the reference frame.
- "Difference" algorithm usually refers to the difference between two adjacent frames where in this case, the previous frame is the reference frame.
- "Ghosting" refers to a second image of the moving object appearing as an artifact of a difference algorithm (since the points where the object was, and where the object is now have both changed between the two frames). Ghosting is especially problematic if the object is moving relatively fast relative to the frame rate of the video.
- "Foreground aperture" refers to a hole appearing in the moving object as an artifact of a difference algorithm.

The difference_with_centroid() function will place a dot in on the image centroid of the difference image, if a difference is found. This function uses a greyscale image as the cv2.moments() function requires the image data only contain a single channel. Although a difference image can just as readily be calculated on RGB data if that's something you want to do!

The double_difference() function calculates the double difference image of the incoming video stream. This is useful for eliminating ghosting and ambiguity that occurs with the naïve image difference approach. 
The double difference approach takes the "bitwise and" of two differences images across 3 frames. This solves the ghosting problem mentioned earlier.

For example, consider the following three "frames" of a "ball" rolling from left to right:
```
Frame 0   Frame 1   Frame 2
|-----|   |-----|   |-----|
|0    |   |  0  |   |    0|
|-----|   |-----|   |-----|
```
A simple difference algorithm with come out with the following two differences (between frames 0&1, and 1&2):
```
Diff 0&1  Diff 1&2
|-----|   |-----|
|0 0  |   |  0 0|
|-----|   |-----|
```
If we take the "bitwise and" of these two difference images (assuming that they're binary images), we get the following:
```
|-----|
|  0  |
|-----|
```
Since only the central "0" is common across the two images. This is how the double difference solves the ghosting issue, albeit at the cost of needing a third frame (and the associated latency) for the algorithm.

### To do:
- Try the `difference()` function by running `differencing.py`.
- In `difference()` add: `ret, diff = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)` 
just before cv2.imshow() to highlight any pixels with a value larger than 24 to white (255), otherwise they become black (0). (This is because a low value such as say 30 may be hard to see but you need to know it exists.)
- Try a lower threshold (for example, 12) and observe the extra salt-and-pepper noise.
- With the lower threshold showing extra noise, add `diff = cv2.medianBlur(diff,3)` after `cv2.threshold()` to see how a median filter can reduce the salt-and-pepper noise.


## Find a Fiducial Marker's 6DOF Pose Information
Fiducial markers are objects that are used to provide a point of reference when the image is processed by a computer vision algorithm. They're commonly scattered around the place for a lot of automatic robotics demonstrations, for example. There are a range of different fiducial markers (you can even use the checkerboard from last lab in a pinch!), but for the purposes of this lab, we will be using the fiducial markers defined in the ArUco library. 

There are a range of ArUco dictionaries that vary in the dimensions of the marker. For this lab, DICT_6X6_250 (each marker is 6x6 bits and the dictionary contains 250 distinct markers). On the next page, the marker with ID 24 can be used to test the code (either printed off or on your computer monitor). Additional markers can be created with the “generate_marker()” function.

Note that on lines 23-27 of `fiducial.py`, the camera matrix and distortion coefficients have been hard-coded from a similar camera to what is being used in the labs. Normally we would load values that have been calculated for the camera being used, as this will give the most accurate results, but for the time being, these values should suffice. If you wish to calculate your own values, please refer to the lab 1 handout where camera calibration is explained.

Additionally, on line 38, the marker has been defined to be 0.035m (3.5cm) wide/tall. For accurate positioning, you should change this to match the height/width of the marker as it appears on your monitor/printout. For the purposes of this lab, however, you're welcome to leave this as it is.

### To do:
- Run `fiducial.py`.
- Find the reliable minimum and maximum limits of distance for x, y and z – and also rotation limits for the three angles.