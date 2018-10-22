from Feature import Feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import glob
from sklearn.svm import LinearSVC
import pickle


def get_image_paths(dataset='cars'):
    """
    Loads image paths for selected dataset.
    :param dataset: Dataset that should be loaded. ['cars', 'notcars']
    :return: Path list 
    """
    path_list = []
    if dataset == 'cars':
        print('Loading data set for \'cars\'...')
        # GTI
        path = 'data/vehicles/GTI*/*.png'
        paths = glob.glob(path)
        print(path)
        print('\t{} elements'.format(len(paths)))
        path_list += paths

        # KITTI
        path = 'data/vehicles/KITTI*/*.png'
        paths = glob.glob(path)
        print(path)
        print('\t{} elements'.format(len(paths)))
        path_list += paths
    elif dataset == 'notcars':
        print('Loading data set for \'notcars\'...')
        # GTI
        path = 'data/non-vehicles/GTI*/*.png'
        paths = glob.glob(path)
        print(path)
        print('\t{} elements'.format(len(paths)))
        path_list += paths

        # Udacity data
        path = 'data/non-vehicles/Extras/*.png'
        paths = glob.glob(path)
        print(path)
        print('\t{} elements'.format(len(paths)))
        path_list += paths

        # Manually extracted data
        path = 'data/non-vehicles/Extracted/*.png'
        paths = glob.glob(path)
        print(path)
        print('\t{} elements'.format(len(paths)))
        path_list += paths
    else:
        raise Exception('There are only two possible choices for c: \'cars\' and \'notcars\'')

    return path_list


def main():
    features = Feature()

    # Load car image paths
    cars = get_image_paths('cars')
    # Load non-car image paths
    notcars = get_image_paths('notcars')

    # Get features for cars
    car_features = features.extract_from_paths(cars)
    # Get features for non-cars
    notcar_features = features.extract_from_paths(notcars)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    # # Set random state to a constant value for debugging
    # rand_state = 42

    # Create train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler (only fir on X_train!)
    X_scaler = StandardScaler().fit(X_train)
    # Save scaler
    with open('scaler.p', 'wb') as scaler_file:
        pickle.dump(X_scaler, scaler_file)

    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Use a linear SVC
    svc = LinearSVC(C=0.5)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Save SVC
    with open('svc.p', 'wb') as svc_file:
        pickle.dump(svc, svc_file)


if __name__ == '__main__':
    main()

    exit(0)
