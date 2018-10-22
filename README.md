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
[project_video_gif]: ./examples/project_output.gif "Project Video GIF"
[project_video]: ./project_video:output.mp4 "Project Video"

---
**Requirements**

* [Anaconda 3](https://www.anaconda.com/download/) is installed on your machine.
* Load dataset for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
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

Optionally you can also add start and/or end times
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

The linear SVC was trained using the `classifier/train.py` script.

The input data, i.e. the extracted features from the image first were scaled using a  StandardScaler from the sklearn.preprocessing library.

```python
from sklearn.preprocessing import StandardScaler

# Initialize and fit StandardScaler
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```

Then the transformed training data was used to train the SVC:

```python
from sklearn.svm import LinearSVC

# Use a linear SVC
svc = LinearSVC(C=0.5)

# Fit classifier
svc.fit(X_train, y_train)
```

Here the parameter C was set to 0.5.
An accuracy score of **0.984** was achieved.


#### Adding more Traning Data

When first training the classifier it seemed to have a lot of false positives on the left kerbside of the lane. This might be caused by a lack of images in the non-vehicle data that contains physical dividers. To improve on this situation, more training data was generated using the `GenerateData.ipynb`. 

Since there is project video car on the left lane of the project video, it is possible to extract images of different sizes from the lower left area of the project video and use those images as non-vehicle training data. It is important to use different sizes, since the classifier will use different sizes (resized to a standard size) as well.

### Sliding Window Search

In the file `utils.py` the following function is contained:
 
```python
def get_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], overlap=(0.5, 0.5),
                window_start_size=(128, 128), window_end_size=(64, 64), layers=1,
                perspective_margin=(75, 75)):
```

It returns all boxes i.e. the segment of the input image extracted, that are later classified. The boxes have different sizes. The image shows, that the blue boxes are the largest and they also cover the larges area. Although the boxes look quite small, they are acutally not. In the image below are only two rows of blue boxes, not three. The two rows overlap by 50 % in x and in y direction. Therefore the boxes appear to be smaller than they are. The same applies to the red and green colored boxes.

![Bounding boxes][bboxes]

### Heatmap

In the file `heatmap.py` the class `Heatmap` is defined. For every detected car, a predefined heat is added to an image. Those heatmaps are also stored for multiple previous frames. This way many false positives can be filtered out since single detections are not enough to create a label.

---

## Video Implementation

Here's a [link to the video result](./project_video_output.mp4)


Here is an example of the debug view. On the left is the output video (labels after cleaning), on the right there are two heatmaps. The top heatmap is the average heatmap over the last stored frame. The lower heatmap is the heatmap of the current frame. The bottom image show all bounding boxes that are classified as cars. 
![Video Output][project_video_gif]

---

## Possible Improvements

* The video still has some false positives. Also the cars are not detected immediately. Tuning the classifier parameters might lead to a more rubust classification
* The bounding boxes are quite unstable. A smoothing algorithm would be useful.
* The framerate is really low. One could think of an optimized algorithm, that only processes every n^th image completely and all the other image only partially
* After all, this "traditional" approach can not keep up (in terms of accuracy and speed) with the YOLO object detection

