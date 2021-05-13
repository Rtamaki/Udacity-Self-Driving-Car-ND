import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import SVM
from scipy.ndimage.measurements import label
import lesson_functions
import random


## Get the direction of the gradients in each pixel
def grad_dir(gray,sobel_kernel=3, ksize=1):
	blur_gray = cv2.GaussianBlur(gray, (1+2*ksize,1+2*ksize), 0)
	abs_sobel_x = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
	abs_sobel_y = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
	sobel_dir = np.arctan2(abs_sobel_y,abs_sobel_x)
	return sobel_dir


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Calculate the gradient direction for the whole image
def img_gradients_direction(img, sobel_kernel=3):

	# For each channel, I calculate the gradients direction
	grad_channel1 = grad_dir(img[:, :, 0],sobel_kernel=sobel_kernel)
	grad_channel2 = grad_dir(img[:, :, 1],sobel_kernel=sobel_kernel)
	grad_channel3 = grad_dir(img[:, :, 2],sobel_kernel=sobel_kernel)


	return grad_channel1, grad_channel2, grad_channel3

def grad_dir_hist(channel1, channel2, channel3, nbins=32, bins_range=(0, 2 * np.pi)):

	# Afterwards each of the resulting image is
	channel1_hist = np.histogram(channel1, bins=nbins, range=bins_range)
	channel2_hist = np.histogram(channel2, bins=nbins, range=bins_range)
	channel3_hist = np.histogram(channel3, bins=nbins, range=bins_range)
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	return hist_features

def bin_grad_dir(channel1, channel2, channel3, size=(32, 32)):

	# Afterwards each of the resulting image is
	grad_img = cv2.resize(np.array([channel1, channel2, channel3]), size).ravel()
	return grad_img


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def alt_extract_features(imgs, color_space='HLS', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        image = image.astype(np.float32)  # they are PNG images so DONT!!!!!!!!!! divide by 255
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV).astype(np.float32)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB).astype(np.float32)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # Not exactly HoG, but the idea behind is still to use thedirection of the gradients, so I will still call iot HoG
        if hog_feat == True:
        	channel1, channel2, channel3 = img_gradients_direction(feature_image, sobel_kernel=3)
        	# grad_hist_features = grad_dir_hist(channel1, channel2, channel3, nbins=hist_bins, bins_range=(0, 256))
        	grad_hist_features = bin_grad_dir(channel1, channel2, channel3, size=spatial_size)
        	# Append the new feature vector to the features list
        	file_features.append(grad_hist_features)
        	features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features





## Same as the lesson function but with the difference in that the HoG is not exactly HoG but the gradients direction in truth
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



def alt_single_img_features(img, grad_channel1, grad_channel2, grad_channel3, 
	color_space='HSV', 
	spatial_size=(32, 32),
	hist_bins=32, 
	orient=9, 
	pix_per_cell=8, 
	cell_per_block=2, 
	hog_channel='ALL',
	spatial_feat=True, 
	hist_feat=True, 
	hog_feat=True):    
    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        # hog_features =grad_dir_hist(grad_channel1, grad_channel2, grad_channel3, nbins=hist_bins)
        hog_features = bin_grad_dir(grad_channel1, grad_channel2, grad_channel3, size=spatial_size)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)





# # Define a function you will pass an image 
# # and the list of windows to be searched (output of slide_windows())
# def alt_search_image(img, classifier, search_config,
#                     hog_channel='ALL', spatial_feat=True, 
#                     hist_feat=True, hog_feat=True):


# 	## Get the configurations for the features
#     clf = classifier.svc
#     scaler = classifier.scaler
#     orient = classifier.orient
#     pix_per_cell = classifier.pix_per_cell
#     cell_per_block = classifier.cell_per_block
#     spatial_size = classifier.spatial_size
#     hist_bins = classifier.hist_bins
#     color_space = classifier.color_space

#     ## Get the configuration for the windos search
#     window_sizes = search_config['window_sizes']
#     search_regions = search_config['search_regions']
#     overlap = search_config['overlap']


#     ## Apply the necessary transformations only once for every kind of possible search configuration
#     draw_image = np.copy(img)
#     img = lesson_functions.convert_color(img.astype(np.float32) / 255.0, conv=color_space)
#     grad_channel1, grad_channel2, grad_channel3 = img_gradients_direction(img)

#     ## Verify if the images sizes have the same size
#     if len(window_sizes) != len(search_regions):
#         print("ERROR: number of search_regions and number of window sizes don't match.")
#         return None
#     else:
#     	windows_list = []
#     	## iterate for each combination of window size and search region
#         for i in range(len(window_sizes)):

#         	windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[search_regions[i][0], search_regions[i][1]],
#                  xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(overlap[i], overlap[i]))

#         	color1 = int(random.uniform(0, 1)*255)
#         	color2 = int(random.uniform(0, 1)*255)
#         	color3 = int(random.uniform(0, 1)*255)
        	


#         	## Search for each window if there SVM finds a car
#         	for window in windows:
#         		## Resize
#         		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
#         		resized_grad_channel1 = cv2.resize(grad_channel1[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
#         		resized_grad_channel2 = cv2.resize(grad_channel2[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
#         		resized_grad_channel3 = cv2.resize(grad_channel3[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

#         		features = alt_single_img_features(test_img, 
# 					resized_grad_channel1,
# 					resized_grad_channel2,
# 					resized_grad_channel3,
# 					color_space=color_space, 
#                             spatial_size=spatial_size, hist_bins=hist_bins, 
#                             orient=orient, pix_per_cell=pix_per_cell, 
#                             cell_per_block=cell_per_block, 
#                             hog_channel='ALL', spatial_feat=True, 
#                             hist_feat=True, hog_feat=True)

# 				#5) Scale extracted features to be fed to classifier
#         		test_features = scaler.transform(np.array(features).reshape(1, -1))
#         		prediction = clf.predict(test_features)
#         		if prediction == 1:
#         			cv2.rectangle(draw_image, window[0], window[1], (color1, color2, color3), 6)
#             		windows_list.append(window)

#         return draw_image, windows_list



# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def alt_search_image(img, classifier, search_config,
                    hog_channel='ALL', spatial_feat=True, 
                    hist_feat=True, hog_feat=True):


	## Get the configurations for the features
    clf = classifier.svc
    scaler = classifier.scaler
    orient = classifier.orient
    pix_per_cell = classifier.pix_per_cell
    cell_per_block = classifier.cell_per_block
    spatial_size = classifier.spatial_size
    hist_bins = classifier.hist_bins
    color_space = classifier.color_space

    ## Get the configuration for the windos search
    window_sizes = search_config['window_sizes']
    search_regions = search_config['search_regions']
    overlap = search_config['overlap']


    ## Apply the necessary transformations only once for every kind of possible search configuration
    draw_image = np.copy(img)
    img = lesson_functions.convert_color(img.astype(np.float32) / 255.0, conv=color_space)
    grad_channel1, grad_channel2, grad_channel3 = img_gradients_direction(img)

    ## Verify if the images sizes have the same size
    if len(window_sizes) != len(search_regions):
        print("ERROR: number of search_regions and number of window sizes don't match.")
        return None
    else:
    	windows_list = []
    	## iterate for each combination of window size and search region
        for i in range(len(window_sizes)):

        	windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[search_regions[i][0], search_regions[i][1]],
                 xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(overlap[i], overlap[i]))

        	color1 = int(random.uniform(0, 1)*255)
        	color2 = int(random.uniform(0, 1)*255)
        	color3 = int(random.uniform(0, 1)*255)
        	


        	## Search for each window if there SVM finds a car
        	for window in windows:
        		## Resize
        		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        		resized_grad_channel1 = cv2.resize(grad_channel1[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        		resized_grad_channel2 = cv2.resize(grad_channel2[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        		resized_grad_channel3 = cv2.resize(grad_channel3[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        		features = alt_single_img_features(test_img, 
					resized_grad_channel1,
					resized_grad_channel2,
					resized_grad_channel3,
					color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel='ALL', spatial_feat=True, 
                            hist_feat=True, hog_feat=True)

				#5) Scale extracted features to be fed to classifier
        		test_features = scaler.transform(np.array(features).reshape(1, -1))
        		prediction = clf.predict(test_features)
        		if prediction == 1:
        			cv2.rectangle(draw_image, window[0], window[1], (color1, color2, color3), 6)
            		windows_list.append(window)

        return draw_image, windows_list











