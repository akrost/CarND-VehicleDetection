import cv2
import numpy as np


def get_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], overlap=(0.5, 0.5),
                window_start_size=(128, 128), window_end_size=(64, 64), layers=1,
                perspective_margin=(75, 75)):
    """
    Calculates all possible windows. The windows become smaller per layer. The window size is linearly interpolated
    between the start and the and size.
    :param img: Input image
    :param x_start_stop: Min and max x position of the area of interest
    :param y_start_stop: Min and max y position of the area of interest
    :param overlap: Relativ overlop of the windows in x and y direction
    :param window_start_size: Start size of the extracted windows
    :param window_end_size: End size of the extracted windows, only used if layers > 1
    :param layers: Number of layers of different sized windows.
    :param perspective_margin: Perspective correction for layers
    :return: List of bounding boxes
    """
    height, width, _ = img.shape

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = width
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = height

    # List of windows / bounding boxes
    windows = []

    # Define append for better performance
    windows_append = windows.append

    # Layers with different sized bounding boxes
    for l in range(layers):
        # For every layer, recalculate current window size
        # Layer 0: window_start_size
        # Layer n: window_end_size
        if l > 0:
            window_size_x = int(window_start_size[0] - l * (window_start_size[0] - window_end_size[0]) / (layers - 1))
            window_size_y = int(window_start_size[1] - l * (window_start_size[1] - window_end_size[1]) / (layers - 1))
        else:
            window_size_x = window_start_size[0]
            window_size_y = window_start_size[1]

        # Adjust x_start, x_stop and y_stop
        x_start = x_start_stop[0] + l * perspective_margin[0]
        x_stop = x_start_stop[1] - l * perspective_margin[0]
        y_stop = y_start_stop[1] - l * perspective_margin[1]

        # Absolute overlap
        overlap_x = int(overlap[0] * window_size_x)
        overlap_y = int(overlap[1] * window_size_y)

        # Rows
        for y in range(y_start_stop[0], y_stop - window_size_y + 1, overlap_y):
            # Columns
            for x in range(x_start, x_stop - window_size_x + 1, overlap_x):
                bbox = ((x, y), (x + window_size_x, y + window_size_y))
                # Append window position to list
                windows_append(bbox)

    # Return the list of windows
    return windows


def get_image_region(img, bbox=((0, 0), (64, 64)), resize=(64, 64)):
    """
    Return a small image that is cropped out of the input image. The returned image is resized to the given size.
    :param img: Input img
    :param bbox: Bounding box which is extracted
    :param resize: Size of the image that is returned
    :return: Resized image section
    """
    crop = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    resized_img = cv2.resize(crop, resize)
    return resized_img


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, offset=(0, 0)):
    """
    Draws bounding boxes on an image.
    :param img: Input image to draw boxes on
    :param bboxes: List of bounding boxes
    :param color: Color of bounding boxes
    :param thick: Line thickness of bounding boxes
    :param offset: Offset added to bounding boxes. Set to top left corner of
    area of interest to draw bounding boxes on non-cropped image
    :return: Input image with bounding boxes drawn on it.
    """
    offset_x, offset_y = offset
    if not offset_x:
        offset_x = 0
    if not offset_y:
        offset_y = 0

    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        p1 = (bbox[0][0] + offset_x, bbox[0][1] + offset_y)
        p2 = (bbox[1][0] + offset_x, bbox[1][1] + offset_y)
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, p1, p2, color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    """
    Draws label bounding boxes on an image.
    :param img: Input image, canvas for drawing boxes on.
    :param labels: (Scipy) Labels to draw on the image
    :return: Input image with bboxes drawn on
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img
