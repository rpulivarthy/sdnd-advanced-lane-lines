## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Introduction 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.jpg "Undistorted"
[image2]: ./output_images/test4_undist.jpg "Undistorted-Raw"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4]: ./output_images/binarty_warped.jpg "Warp Example"
[image5]: ./output_images/warp_binary_roi.jpg "Region of interest"
[image6]: ./output_images/lanes.jpg "Lane Lines"
[image7]: ./output_images/lane_detected.jpg "Projected lines"
[video1]: ./processed_project_video.mp4 "Video"

I will explain each goal and the steps taken to acheive it.

## Camera Calibration
The images taken from the camera are distorted by the very nature of how cameras work. It's important to remove the distortion
to calculate the parameters more precisely. In the project the camera that took the video is the same camera that was used for test images
and calibration pictures. The images used for calibration are of different shapes. I have used a range of nx and ny values to plot the corners
in the images. All the objects points compared with the image points and using cv2 functions i wass able to calculate the distortion parameters.
The parameters are stored for use furthur in the pipeline

![alt text][./camera_cal/calibration1.jpg] ![alt text][image1]

## Apply a distortion correction to raw images
Applying the undistortion to one of the test images
![alt text][image2]

## Creating a thresholded binary image
For thersholding I have used the HLS space. I have sobel operator to identity the lines. Using the gray scale loses the color information.
I have used a combination of thresholds x- gradient, saturation and luminosity to get the final binary threshold.
The binary threshold and the image with color channels is below.
![alt text][image3]

## Perspective transformation
Perspective transformation is needed to get the exact oreintation of the lane lines. The lane lines will give a impression the left and right lines merge; though that never happens. Perspective transform gives the bird view of the lines, which are useful to predict the lanes. Snippet from code.

```
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])    
```
![alt text][image4]
![alt text][image5]

## Detect Lanes
I masked the images from perspective transform to region of interest to remove the unnecessary information and to plot the lanes better.
Divide the image to left half and right half. Look for the maximum number of pixels in each off. Give the margin and draw a boundary.
Keep moving the boundary upwards to the end of the image. 

Once one of the first frames is processed, I use the last known line location to restrict my search for new lane pixels.

![alt text][image6]

## Extracting the local curvature of the road and vehicle localization

The radius of curvature is computed upon calling the `Line.update_fits` method of a line. The method that does the computation is called `Line.get_radius_of_curvature()`. The mathematics involved is summarized in [this tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).  
For a second order polynomial f(y)=A y^2 +B y + C the radius of curvature is given by R = [(1+(2 Ay +B)^2 )^3/2]/|2A|.

The distance from the center of the lane is computed in the `line.get_position_from_center` method, which essentially measures the distance to each lane and computes the position assuming the lane has a given fixed width of 3.7m. 

## Projecting the detected lane lines


![alt text][image7]


---

# Video Processing Pipeline

Finally, I took the code developed and processed the project video in the notebook  `stage2_video_pipeline.ipynb`. The processed project video can be found here:
[link to my video result](./processed_project_video.mp4)
