import SVM
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import lesson_functions
import features_extraction
import cv2



def examine_windows(image, svm, window_boxes):
	# This function examines each window with the SVM
	# and gives back which windows are classified as cars
	cars_windows_list = []
	for (point1, point2) in window_boxes:
		window_img = image[point1[1]:point2[1] ,point1[0]:point2[0]]
		resized_window = cv2.resize(window_img, (64, 64)) / 255.0
		classification = SVM.get_prediction(svm, resized_window)
		# if classification == 1:
		# 	cars_windows_list.append((point11, point2))

	return cars_windows_list



img_name = 'test3.jpg'
file = './test_images/' + img_name
svc, acc, scaler = SVM.get_classifier()
image = cv2.imread(file) 
# test_image = cv2.imread("./vehicles/GTI_Left/image0015.png") /255.0

# windows = image[0:64, 0:64]
# print(SVM.get_prediction(svc, windows))

window_size = (64, 64)
y_start_stop = [image.shape[0]/2, image.shape[0]]
window_boxes = lesson_functions.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                 xy_window=window_size, xy_overlap=(0.5, 0.5))

print(examine_windows(image, svc, window_boxes))



# print(test_image.shape)
# windows = image[0:64, 0:64]
# print(windows.shape)
# print(SVM.get_prediction(svc, windows))

# cv2.imshow('fds', windows)
# cv2.waitKey(5000)

# SVM.test_classifier()

# window_y_limits = [int(image.shape[0]/3*2), int(image.shape[0]/4*3), int(image.shape[0]),  int(image.shape[0])]


# drawed_img = lesson_functions.draw_boxes(image, window_boxes)


# print(window_boxes[0])
# drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_RGB2BGR)
# car_boxes = examine_windows(image, svc, window_boxes)
# cv2.imwrite('./output_images/' + img_name, drawed_img)
# cv2.imshow('fds', drawed_img)
# cv2.waitKey(5000)



