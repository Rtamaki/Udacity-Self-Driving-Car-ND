import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split # using sklearn >=0.18
import matplotlib.image as mpimg

import time
import lesson_functions
import alternative_features_calculation
import os
import pickle


def get_training_data_files(car_folder, not_car_folder):
    list = os.listdir(car_folder)
    car_images = []
    for inner_folders in list:
        images = os.listdir(car_folder + '/' + inner_folders)
        for img_name in images:
            if img_name != '.DS_Store':
                car_images.append(car_folder + '/' + inner_folders + '/' + img_name)

    list = os.listdir(not_car_folder)
    not_car_images = []
    for inner_folders in list:
        images = os.listdir(not_car_folder + '/' + inner_folders)
        for img_name in images:
            if img_name != '.DS_Store':
                not_car_images.append(not_car_folder + '/' + inner_folders + '/' + img_name)

    return car_images, not_car_images



def train_SVM():
    svc = LinearSVC()
    car_images, not_car_images = get_training_data_files('./vehicles', './non-vehicles')
    car_features = lesson_functions.extract_features(car_images)
    notcar_features = lesson_functions.extract_features(not_car_images)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    training_time = t2 - t1
    acc = round(svc.score(X_test, y_test), 4)
    print ('Training time in seconds: ', round(training_time, 5))
    print('Accuracy was: ', round(acc, 5))
    return svc, acc, training_time, X_scaler

def get_classifier():
    try:
        svc = pickle.load(open('svc_model.sav', 'rb'))
        acc = pickle.load(open('acc.sav', 'rb'))
        scaler = pickle.load(open('scaler.sav', 'rb'))

    except IOError:
        svc, acc, training_time, scaler = train_SVM()
        pickle.dump(svc, open('svc_model.sav', 'wb'))
        pickle.dump(acc, open('acc.sav', 'wb'))
        pickle.dump(scaler, open('scaler.sav', 'wb'))
    return svc, acc, scaler

# for a given image, get the prediction from the classifier

def get_prediction(svc, img):
    features = lesson_functions.get_features(img)
    return svc.predict([features])

def test_classifier():
    svc, acc = get_classifier()
    vehicle_img_name = './vehicles/GTI_Far/image0041.png'
    vehicle_img = mpimg.imread(vehicle_img_name)

    not_vehicle_img_name = './non-vehicles/GTI/image26.png'
    not_vehicle_img = mpimg.imread(not_vehicle_img_name)

    print(get_prediction(svc, vehicle_img))
    print(get_prediction(svc, not_vehicle_img))



class SVM_classifier():

    """docstring for SVM_classifier"""
    def __init__(self, 
        color_space='YCrCb', 
        HoG=0, 
        spatial_size = (32, 32), 
        hist_bins = 32, 
        orient=9, 
        pix_per_cell=16, 
        cell_per_block=2,
        method='classic'):
        
        self.color_space = color_space
        self.HoG = HoG
        self.svc = None
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell=pix_per_cell
        self.cell_per_block=cell_per_block
        self.method=method
        
        self.get_classifier()

    #ge configurations from the inputs given
    def get_config(self):
        self.config = {'color_space': self.color_space, 
        'HoG': self.HoG, 
        'spatial_size': self.spatial_size, 
        'hist_bins': self.hist_bins, 
        'orient': self.orient,
        'pix_per_cell': self.pix_per_cell,
        'cell_per_block': self.cell_per_block,
        'method': self.method}

    def set_config(self):
        self.color_space = self.config['color_space']
        self.HoG = self.config['HoG']
        self.spatial_size = self.config['spatial_size']
        self.hist_bins = self.config['hist_bins']
        self.orient = self.config['orient']
        self.pix_per_cell=self.config['pix_per_cell']
        self.cell_per_block=self.config['cell_per_block']
        self.method = self.config['method']


    def get_classifier(self):
        try:
            self.svc = pickle.load(open('svc_model.sav', 'rb'))
            self.acc = pickle.load(open('acc.sav', 'rb'))
            self.scaler = pickle.load(open('scaler.sav', 'rb'))
            self.config = pickle.load(open('model_config.sav', 'rb'))
            self.set_config()

        except IOError:
            self.svc, self.acc, training_time, self.scaler = self.train_SVM()
            self.get_config()
            pickle.dump(self.config, open('model_config.sav', 'wb'))
            pickle.dump(self.svc, open('svc_model.sav', 'wb'))
            pickle.dump(self.acc, open('acc.sav', 'wb'))
            pickle.dump(self.scaler, open('scaler.sav', 'wb'))
        print("Loaded classifier")


    def train_SVM(self):
        svc = LinearSVC()
        car_images, not_car_images = get_training_data_files('./vehicles', './non-vehicles')
        t1 = time.time()
        if self.method == 'classic':
            print("Using classical method")
            car_features = lesson_functions.extract_features(car_images, 
                color_space=self.color_space, 
                hog_channel=self.HoG,
                orient=self.orient,
                spatial_size=self.spatial_size,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hist_bins=self.hist_bins)
            notcar_features = lesson_functions.extract_features(not_car_images, 
                color_space=self.color_space, 
                hog_channel=self.HoG,
                orient=self.orient,
                spatial_size=self.spatial_size,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hist_bins=self.hist_bins)
        else:
            print("Using alternative method")
            car_features = alternative_features_calculation.alt_extract_features(car_images, 
                color_space=self.color_space, 
                hog_channel=self.HoG,
                orient=self.orient,
                spatial_size=self.spatial_size,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hist_bins=self.hist_bins)

            notcar_features = alternative_features_calculation.alt_extract_features(not_car_images, 
                color_space=self.color_space, 
                hog_channel=self.HoG,
                orient=self.orient,
                spatial_size=self.spatial_size,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hist_bins=self.hist_bins)
        t2 = time.time()
        training_time = t2 - t1
        print ('Feature processing time in seconds: ', round(training_time, 5))

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        t1 = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        training_time = t2 - t1
        acc = round(svc.score(X_test, y_test), 4)
        print ('Training time in seconds: ', round(training_time, 5))
        print('Accuracy was: ', round(acc, 5))
        return svc, acc, training_time, X_scaler

    def get_prediction(self, img):
        features = lesson_functions.single_img_features(img, 
            color_space=color_space, 
            hog_channel=HoG,
            orient=self.orient,
            spatial_size=self.spatial_size,
            pix_per_cell=self.pix_per_cell,
            cell_per_block=self.cell_per_block,
            hist_bins=self.hist_bins)
        return (self.svc).predict([features])
        
    def get_bboxes(self, img):
        if self.method == 'classic':

            window_sizes = [1.0, 1.5, 2.0]
            search_regions = [[420, 520], [400, 560], [400, 600]]
            overlap = [0.5, 0.5]

            search_config = {'window_sizes': window_sizes, 'search_regions': search_regions, 'overlap': overlap}
            x, bboxes = lesson_functions.search_image(img, self, search_config, draw=True)

            
        else:
            window_sizes = [64, 96, 128, 256]
            search_regions = [[420, 560], [400, 520], [400, 600], [420, 720]]
            overlap = [0.25, 0.25, 0.25, 0.125]
            search_config = {'window_sizes': window_sizes, 'search_regions': search_regions, 'overlap': overlap}
            x, bboxes = alternative_features_calculation.alt_search_image(img, self, search_config)

        return x, bboxes
