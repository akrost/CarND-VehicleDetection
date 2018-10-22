import pickle
import numpy as np
import cv2
import argparse
from configparser import ConfigParser
from moviepy.editor import VideoFileClip

from classifier.Feature import Feature
from Heatmap import Heatmap
from utils import *

# Load X_Scaler
with open('classifier/scaler.p', 'rb') as scaler_file:
    X_scaler = pickle.load(scaler_file)

# Load SVC
with open('classifier/svc.p', 'rb') as svc_file:
    svc = pickle.load(svc_file)

# Load config file
conf_path = './conf/config.cfg'
config = ConfigParser()
config.read(conf_path)

# Load Sliding Window variables
sw_section = 'SlidingWindow'
top_left_x = config.getint(sw_section, 'top_left_x')
top_left_y = config.getint(sw_section, 'top_left_y')
bottom_right_x = config.getint(sw_section, 'bottom_right_x')
bottom_right_y = config.getint(sw_section, 'bottom_right_y')
start_win_width = config.getint(sw_section, 'start_win_width')
start_win_height = config.getint(sw_section, 'start_win_height')
end_win_width = config.getint(sw_section, 'end_win_width')
end_win_height = config.getint(sw_section, 'end_win_height')
overlap_frac_x = config.getfloat(sw_section, 'overlap_frac_x')
overlap_frac_y = config.getfloat(sw_section, 'overlap_frac_y')
layer = config.getint(sw_section, 'layer')

# Initialize Feature
feature = Feature()

# Initialize heatmap
heatmap = Heatmap()


def load_img(path):
    """
    Load image from path.
    :param path: Path to image
    :return: Image in BGR
    """
    img = cv2.imread(path)
    return img


def process_image(img, debug=True):
    """
    Processes video frames or RGB image.
    1) Find image segments to classify. This part is currently static but also might be changed to a non-static approach
    (e.g. create area of interest based on lane curvature.
    2) Resize every image from 1) and extract feature vector.
    3) Scale feature vector using pre trained scaler
    4) Predict class for every image segment
    5) If segment is classified as a car, add heat to the heat map at its position
    6) Extract labels based on heat map.
    :param img: Video frame or image
    :param debug: If True, output image is extended by debugging view.
    :return: Output image/frame
    """
    global X_scaler, svc, feature, heatmap

    global top_left_x, top_left_y, bottom_right_x, bottom_right_y
    global start_win_width, start_win_height, end_win_width, end_win_height
    global overlap_frac_x, overlap_frac_y, layer

    out_img = np.copy(img)

    # When training the model, the images are loaded by cv2 => BGR (they might be converted later, though)
    # Since moviepy read images as RGB, we need to convert them to BGR first
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    windows = get_windows(img,
                          x_start_stop=[top_left_x, bottom_right_x],
                          y_start_stop=[top_left_y, bottom_right_y],
                          overlap=(overlap_frac_x, overlap_frac_y),
                          window_start_size=(start_win_width, start_win_height),
                          window_end_size=(end_win_width, end_win_height),
                          layers=layer)

    car_windows = []
    for window in windows:
        # Extract bounding box image from frame
        cropped_img = get_image_region(img, bbox=window)

        # Get feature vector
        feature_vector = feature.extract(cropped_img).astype(np.float64)

        # Normalize vector
        scaled_feature_vetor = X_scaler.transform(feature_vector)

        # Make prediction
        pred = svc.predict(scaled_feature_vetor)

        # If pred[0] == 1. then a car was detected
        if pred[0] == 1.:
            car_windows.append(window)

    # Add heat to heatmap where cars were detected
    heatmap.add_heat(car_windows)

    # Get labels from heatmap
    l = heatmap.get_labels()

    # Create image with all detected labels (post-heatmap)
    label_img = draw_labeled_bboxes(out_img, l)

    if debug:
        print('cars found: {}'.format(l[1]))

        # Create image with all detected cars (pre-heatmap)
        box_img = draw_boxes(out_img, car_windows)

        # Create image that is an average of the last frames heatmap
        last_heatmaps_img = heatmap.last_maps_average()

        # Reduce size to 1/3
        small_last_heatmaps_img = cv2.resize(last_heatmaps_img,
                                             (last_heatmaps_img.shape[1]//3, last_heatmaps_img.shape[0]//3))
        small_current_heatmap_img = cv2.resize(heatmap.current_map, (heatmap.shape[1]//3, heatmap.shape[0]//3))
        small_box_img = cv2.resize(box_img, (box_img.shape[1]//3, box_img.shape[0]//3))

        # Create debug view
        right_img = np.vstack((small_last_heatmaps_img, small_current_heatmap_img, small_box_img))

        # Add debug view to video
        out_img = np.hstack((label_img,
                             right_img))
    else:
        out_img = np.copy(label_img)

    # Move current heatmap to archive, create new map for next frame
    heatmap.next_map()

    return out_img


def main(input_video, output_video, t_start, t_end):
    clip_in = VideoFileClip(input_video).subclip(t_start, t_end)
    clip = clip_in.fl_image(process_image)
    clip.write_videofile(output_video, audio=False)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Script will detect cars in a given video.')
    parser.add_argument('input_video', help='Path to video clip')
    parser.add_argument('output_video', help='Path to output video clip')
    parser.add_argument('--t_start', help='Start time of video processing. Format \'01:03:05.3\'', default=0)
    parser.add_argument('--t_end', help='End time of video processing. Format \'01:03:05.3\'', default=None)
    args = parser.parse_args()

    main(input_video=args.input_video,
         output_video=args.output_video,
         t_start=args.t_start,
         t_end=args.t_end)

    exit(0)
