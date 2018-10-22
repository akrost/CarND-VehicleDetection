# **Vehicle Detection**

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[bboxes]: ./examples/bounding_boxes.png "Bounding Boxes"
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
**Requirements**

* [Anaconda 3](https://www.anaconda.com/download/) is installed on your machine.
* Load data set for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
---
## **Getting started**

1. Clone repository:<br/>
```sh
git clone https://github.com/akrost/CarND-VehicleDetection.git
cd carnd-vehicledetection
```

2. Create and activate Anaconda environment:
```sh
conda create --name carnd-p5 python=3.6
source activate carnd-p5
```
Activating the environment may vary for your OS.

3. Install packages:
```sh
pip install -r requirements.txt
```

4. Run the prediction
```sh
python main.py project_video.mp4 project_video_output.mp4
```

Optionally you can also add a start and/or end times
```sh
python main.py project_video.mp4 project_video_output.mp4 --t_start '00:00:05' --t_end '00:00:23.234'
```

or just check out the help:

```sh
python main.py -h
usage: main.py [-h] [--t_start T_START] [--t_end T_END]
               input_video output_video

Script will detect cars in a given video.

positional arguments:
  input_video        Path to video clip
  output_video       Path to output video clip

optional arguments:
  -h, --help         show this help message and exit
  --t_start T_START  Start time of video processing. Format '01:03:05.3'
  --t_end T_END      End time of video processing. Format '01:03:05.3'
```

If you want to train the model on your own, you have to update the function `get_image_paths()` in the file `train.py` to match your file structure. Running the pretrained models should work without any changes, though. 


---
## **Project**

### Data exploration

The project was started with a brief data exploration. It can be found in the Jupyter Notebook `classifier/ExploreData.ipynb`. This notebook mainly looks at the count of images per class and visualizes a few exampel images. The key finding is that the GTI data is less diverse than the KITTI dataset is. 

### Features

The class `Feature` in the file `Feature.py` is used to set the parameters for the feature extraction. The Feature class has three children, `BinSpatial(Feature)`, `ColorHist(Feature)` and `HOG(Feature)`. In the Feature class those three types of features can be added (`.add()`) to the list of features. Using the `.extract()` method, those feactures are extracted from an image. To train the classifier efficiently, there is another method called `.extract_from_paths()`. It's basically the same as the .etract() method, but instead of feading an image into it, it is provided with a list of paths. 

#### Spatial Binning of Color

The `BinSpatial(Feature)` class uses spatial binning of color to generate a feature vector. It can be done for various color spaces an any image size. The size of the extracted feature vector directly correlates to the size of the image. The `extract_feature()` method convert the color space, resizes the input image to the given size and then flattens the image. The flattened image can be used as a color-aware feature. 


Final choice of parameters:

|Parameter | Value |
|:---------|:------|
|Color space | LUV |
|Spatial size | (32, 32) |


#### Color Histogram

The `ColorHist(Feature)` class creates a histogram of the colors of the given image. It takes the parameter *nbins* and *bins_range*. *nbins* describes into how many equal-sized bins the histogram is divided. *bins_range* describes the lower and upper limit for pixel values that are considered.   

Final choice of parameters:
Color histogram was not used.

#### Histogram of Oriented Gradients (HOG)

The `HOG(Feature)` class creates a histogram of oriented gradients of the input image. One can choose between various color spaces (*cspace*) and whether to use one specific or all channels (*hog_channel*). Also the number of orientations (*orient*), the pixels per cell (*pix_per_cell*) and the cells per block (*cell_per_block*) can be tuned. The class also has the `.extract_feature()` method, that returns the HOG feature vector.

Final choice of parameters

| Parameter  | Value |
|:-----------|:------|
| Color space | YCrCb |
| orient     | 9     |
| pix_per_cell | 8   |
| cell_per_block | 2 |
| hog_channel | 'ALL' |


### Training the Classifier
I trained a linear SVM using...

#### Adding more Traning Data

### Sliding Window Search

In the file `utils.py` the following function is defined:

```python
def get_windows(img, x_start_stop=[200, 1280], y_start_stop=[400, 625], overlap=(0.5, 0.5),
                window_start_size=(128, 128), window_end_size=(64, 64), layers=3,
                perspective_margin=(75, 75))
```

It returns all sections of the frame/image that are later classified. The picture below shows the grid of windows that are returned. Note: The boxes appear to be smaller than they actually are, because they are overlapping 50% both in x and y direction. The biggest (blue) boxes cover the biggest area of the region of interest, whereas the smalles (red) boxes only cover the horizon of the streets.

![Bounding boxes][bboxes]

#### Examples

Ultimately I searched on thre scales using YCrCb 3-channel HOG features and spatially binned color (LUV) in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

