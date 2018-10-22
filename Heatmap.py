import numpy as np
import cv2
from scipy.ndimage.measurements import label


class Heatmap:
    """
    Class to handle heat map.
    """
    def __init__(self, shape=(720, 1280, 3), num_last_saved=5, last_map_influence=.2):
        self.HEAT = 25

        self.shape = shape
        self.current_map = self._initialize_heatmap()
        self.last_maps = []
        self.num_last_saved = num_last_saved
        self.last_map_influence = last_map_influence

    def _initialize_heatmap(self):
        """
        Initializes a new heat map
        :return: New heat map
        """
        return np.zeros(shape=self.shape).astype(np.uint8)

    @staticmethod
    def _apply_threshold(heatmap, threshold):
        """
        Applies threshold to a heat map. All pixels of the heat map that are below the threshold are set to zero.
        :param heatmap: Heat map to apply the threshold to
        :param threshold: Threshold [0..255]
        :return: Thresholded heat map
        """
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def add_heat(self, bbox_list):
        """
        Add heat to the current heat map at the position of the given bounding boxes.
        :param bbox_list: List of bounding boxes. Format: ((x1, y1), (x2, y2))
        :return: Current heat map
        """
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.current_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.HEAT

        # Return current heatmap
        return self.current_map

    def next_map(self):
        """
        Stores current heat map in the archive and re-initializes the current map for the next frame.
        :return: -
        """
        # Remove oldest map from last maps, if too many maps stored
        if len(self.last_maps) >= self.num_last_saved:
            self.last_maps = self.last_maps[1:]

        # Add current map to last maps
        self.last_maps.append(self.current_map)

        # Reset current map
        self.current_map = self._initialize_heatmap()

    def last_maps_average(self):
        """
        Calculates an average over all stored archive heat maps. All stored heat maps are added with the same weight
        :return: Average of archive maps
        """
        average_last_map = self._initialize_heatmap()

        for m in self.last_maps:
            average_last_map = cv2.addWeighted(average_last_map, 1, m, self.last_map_influence, 0)

        return average_last_map

    def get_labels(self, threshold=40):
        """
        Get labels for current heat map (weight=0.8) and average of archive maps (weight=1). Apply threshold to heat map
        before extracting labels
        :param threshold: Threshold to apply
        :return: (Scipy) Labels of heat map
        """
        average_last_map = self.last_maps_average()
        average_last_map = cv2.addWeighted(average_last_map, 1, self.current_map, 0.8, 0)

        average_last_map = self._apply_threshold(average_last_map, threshold)

        return label(average_last_map)
