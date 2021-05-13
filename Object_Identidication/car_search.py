import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
import SVM
import alternative_features_calculation
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label






image = (mpimg.imread('./test_images/test4.jpg')).astype(np.float32) 
draw_image = np.copy(image)
classifier = SVM.SVM_classifier(method='classic')        


window_sizes = [64, 80, 96, 128]
search_regions = [[360, 460], [400, 520], [520, 720], [420, 720]]
overlap = [0.25, 0.25, 0.25, 0.25]




# window_sizes = [64, 96]

# search_regions = [[360, 660], [420, 720]]
# overlap = [0.5, 0.25]
search_config = {'window_sizes': window_sizes, 'search_regions': search_regions, 'overlap': overlap}
  

# x, bboxes = search_image(image, classifier, search_config, draw=True)
# x, bboxes = alt_find_cars(image, 360, 720, 
#     search_config,
#     classifier, 
#     window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     )
# x, bboxes = alternative_features_calculation.alt_search_image(image, classifier, search_config)
x, bboxes = classifier.get_bboxes(image)

# plt.imshow(window_img)
img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
cv2.imshow('fjbf', img)
cv2.imwrite( 'test.jpg', img)

cv2.waitKey(5000)


