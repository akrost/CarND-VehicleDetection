import cv2
import numpy as np
from skimage.feature import hog


class Feature:
    def __init__(self):
        """
        Initialize feature list. This class is used both to train the model and to predict a class.
        """
        self.feature_list = []
        luv_binning = BinSpatial(color_space='LUV')
        # c_hist = ColorHist()
        # luv_hog = HOG('LUV', orient=12, pix_per_cell=8, cell_per_block=2, hog_channel=0)
        ycrcb_hog = HOG('YCrCb')

        self.add(luv_binning)
        # self.add(c_hist)
        # self.add(luv_hog)
        self.add(ycrcb_hog)

    def add(self, feature):
        """
        Adds feature to feature list.
        :param feature: Feature to add to list
        :return: -
        """
        self.feature_list.append(feature)

    def extract_from_paths(self, img_paths):
        """
        Extracts all features for all images given. If multiple features are added, the individual feature vectors are
        concatenated.
        :param img_paths: Array of image paths.
        :return: Array of features.
        """
        features = []
        for path in img_paths:
            extracted = []
            img = cv2.imread(path)

            for feature in self.feature_list:
                extracted.append(feature.extract_feature(img))

            extracted = np.concatenate(extracted)
            features.append(extracted)
        return np.array(features)

    def extract(self, img):
        """
        Extracts all features for a given image.
        :param img: Input image
        :return: Features
        """
        features = []
        for feature in self.feature_list:

            features.append(feature.extract_feature(img))

        features = np.concatenate(features)
        return np.array([features])


class BinSpatial(Feature):
    """
    Feature class for Spatial Binning of Color
    """
    def __init__(self, color_space='BGR', spatial_size=(32, 32)):
        """
        Initialize feature specific parameters.
        :param color_space: Color space to use for spatial binning
        :param spatial_size: Sets number of bins (e.g. (32, 32) with 3 channels results in 32x32x3=3,072 bins)
        """
        self.color_space = color_space
        self.spatial_size = spatial_size

        self.vector = None

    def extract_feature(self, img):
        """
        Extracts spatial binning feature from image.
        :param img: Input image
        :return: Features
        """
        # Convert image to new color space (if specified)
        if self.color_space != 'BGR':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            else:
                raise Exception('Unsupported color space! Choose \'BGR\', \'HSV\', \'LUV\', \'HLS\', \'YUV\' or'
                                '\'YCrCb\'!')
        else:
            feature_image = np.copy(img)
        # Use .ravel() to create the feature vector
        features = cv2.resize(feature_image, self.spatial_size).ravel()
        # Set the feature vector
        self.vector = features

        return features


class ColorHist(Feature):
    """
    Feature class for Color Histogram
    """
    def __init__(self, nbins=32, bins_range=(0, 256)):
        """
        Initialize feature specific parameters.
        :param nbins: Number of equal-width bins in the given range
        :param bins_range: Lower and upper range of the bins
        """
        self._num_bins = nbins
        self._bins_range = bins_range

        self.vector = None

    def extract_feature(self, img):
        """
        Extracts color histogram from image.
        :param img: Input image.
        :return: Features
        """
        # Compute the histogram of the color channels separately
        channel0_hist = np.histogram(img[:, :, 0], bins=self._num_bins, range=self._bins_range)
        channel1_hist = np.histogram(img[:, :, 1], bins=self._num_bins, range=self._bins_range)
        channel2_hist = np.histogram(img[:, :, 2], bins=self._num_bins, range=self._bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel0_hist[0], channel1_hist[0], channel2_hist[0]))

        self.vector = hist_features
        # Return the individual histograms, bin_centers and feature vector
        return hist_features


class HOG(Feature):
    """
    Feature class for HOG (Histogram of Oriented Gradients)
    """
    def __init__(self, cspace='BGR', orient=9, pix_per_cell=8, cell_per_block=2,
                 hog_channel='ALL'):
        """
        Initialize feature specific parameters.
        :param cspace: Color space to use for HOG
        :param orient: Number of orientations
        :param pix_per_cell: Pixels per cell
        :param cell_per_block: Cells per block
        :param hog_channel: HOG channel ["ALL", 0, 1, 2]
        """
        self._cspace = cspace
        self._orient = orient
        self._pix_per_cell = pix_per_cell
        self._cell_per_block = cell_per_block
        self._hog_channel = hog_channel

        self.vector = None

    @staticmethod
    def _get_hog_features(img, orient, pix_per_cell, cell_per_block):
        """
        Wrapper function for skimage.feature.hog
        :param img: Input image
        :param orient: Number of orientations
        :param pix_per_cell: Pixels per cell
        :param cell_per_block: Cells per block
        :return: HOG-Features
        """
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       feature_vector=True)
        return features

    def extract_feature(self, img):
        """
        Extracts HOG features from input image.
        :param img: Input image
        :return: Features
        """
        # apply color conversion if other than 'RGB'
        if self._cspace != 'BGR':
            if self._cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self._cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self._cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self._cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self._cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            else:
                raise Exception('Unsupported color space! Choose \'BGR\', \'HSV\', \'LUV\', \'HLS\', \'YUV\' or'
                                '\'YCrCb\'!')
        else:
            feature_image = np.copy(img)

        # Call get_hog_features() with vis=False, feature_vec=True
        if self._hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self._get_hog_features(feature_image[:, :, channel],
                                                           self._orient, self._pix_per_cell,
                                                           self._cell_per_block))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self._get_hog_features(feature_image[:, :, self._hog_channel],
                                                  self._orient, self._pix_per_cell,
                                                  self._cell_per_block)

        self.vector = hog_features

        # Return list of feature vectors
        return hog_features
