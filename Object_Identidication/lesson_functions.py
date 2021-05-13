import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import SVM
from scipy.ndimage.measurements import label
import time





# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)): # before it was [0,256]
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='HLS', spatial_size=(32, 32),
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
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
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


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



def single_img_features(img, color_space='HSV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    else: feature_image = np.copy(img)      
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
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)       
        else:
            hog_features = [get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)]
            hog_features = np.ravel(hog_features)  
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='HSV', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    



def convert_color(img, conv='YCrCb'):
    if conv == 'RGB':
        return img
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB).astype(np.float32) 
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV).astype(np.float32) 
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) 
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32) 


# def find_cars(img, ystart, ystop, scale, classifier, 
#     window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     ):
    
#     svc = classifier.svc
#     X_scaler = classifier.scaler
#     orient = classifier.orient
#     pix_per_cell = classifier.pix_per_cell
#     cell_per_block = classifier.cell_per_block
#     spatial_size = classifier.spatial_size
#     hist_bins = classifier.hist_bins

#     draw_img = np.copy(img)
#     img = img.astype(np.float32)/255.0
    
#     img_tosearch = img[ystart:ystop,:,:] 
#     # ctrans_tosearch = (cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCR_CB)).astype(np.float32)
#     # ctrans_tosearch = (cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)).astype(np.float32) 
#     ctrans_tosearch = convert_color(img_tosearch, classifier.color_space)
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
#     ch1 = ctrans_tosearch[:,:,0]
#     ch2 = ctrans_tosearch[:,:,1]
#     ch3 = ctrans_tosearch[:,:,2]

#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
#     nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
#     nfeat_per_block = orient*cell_per_block**2
    
    
#     nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
#     # Compute individual channel HOG features for the entire image


#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

#     windows_list = []
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb*cells_per_step
#             xpos = xb*cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

#             xleft = xpos*pix_per_cell
#             ytop = ypos*pix_per_cell

#             # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
#             # Get color features
#             spatial_features = bin_spatial(subimg, size=spatial_size)
#             hist_features = color_hist(subimg, nbins=hist_bins)

#             x = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

#             # Scale features and make a prediction
#             test_features = X_scaler.transform(x)    
#             #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
#             test_prediction = svc.predict(test_features)
#             if test_prediction == 1:
#                 xbox_left = np.int(xleft*scale)
#                 ytop_draw = np.int(ytop*scale)
#                 win_draw = np.int(window*scale)
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
#                 windows_list.append([ (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]) 
                
#     return draw_img, windows_list
def find_cars(img, ystart, ystop, scale, classifier, 
    window = 64 # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    ):
    
    svc = classifier.svc
    X_scaler = classifier.scaler
    orient = classifier.orient
    pix_per_cell = classifier.pix_per_cell
    cell_per_block = classifier.cell_per_block
    spatial_size = classifier.spatial_size
    hist_bins = classifier.hist_bins

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255.0
    
    img_tosearch = img[ystart:ystop,:,:] 
    # ctrans_tosearch = (cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCR_CB)).astype(np.float32)
    # ctrans_tosearch = (cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)).astype(np.float32) 
    ctrans_tosearch = convert_color(img_tosearch, classifier.color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    x_start = 300

    # Define blocks and steps as above
    nxblocks = ((ch1.shape[1] - x_start) // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image


    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    # hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    # hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)




    windows_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step + x_start // pix_per_cell
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            # hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            # hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            # hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # t1 = time.time()
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # t2 = time.time()
            # training_time = t2 - t1
            # print ('Normal processing time in seconds: ', round(training_time, 5))



            x = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(x) 

            # t3 = time.time()
            # training_time = t3 - t2
            # print ('Scaling processing time in seconds: ', round(training_time, 5))   
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)


            # t4 = time.time()
            # training_time = t4 - t3
            # print ('Prediction processing time in seconds: ', round(training_time, 5))

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                windows_list.append([ (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]) 

            # t5 = time.time()
            # training_time = t5 - t4
            # print ('Drawing processing time in seconds: ', round(training_time, 5))
            # print("--------------------------------------------------")



    return draw_img, windows_list







# This function takes the SVM_classifier and search for vehicles
# for a given number of windows sizes and the respective search region for each window size
def detect_vehicle(SVM_classifier, image, window_sizes, search_regions):

    if len(window_sizes) != len(search_regions):
        print("Different amount of window sizes and search regions. Aborted image search.")
        return None
    else:
        windows_positions = np.array([])
        for i in range(len(window_sizes)):
            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=search_regions[i], 
                    xy_window=window_sizes[i], xy_overlap=(0.75, 0.75))

            hot_windows = search_windows(image, windows, 
                SVM_classifier.svc,
                 SVM_classifier.scaler, 
                 color_space=SVM_classifier.color_space, 
                 spatial_size=SVM_classifier.spatial_size, 
                 hist_bins=SVM_classifier.hist_bins, 
                 orient=9,
                  pix_per_cell=8, 
                  cell_per_block=2, 
                   hog_channel=SVM_classifier.HoG, 
                    spatial_feat=True, 
                    hist_feat=True, 
                    hog_feat=True)

            if len(hot_windows) > 0 and type(hot_windows) != None:
                windows_positions = np.vstack((windows_positions, hot_windows))
                print(hot_windows[0])

    return windows_positions




# Search the image for cars by using different configurations of window sizes and search regions
def search_image(image, classifier, search_config, draw=False):
    
    img = np.copy(image)
    window_sizes = search_config['window_sizes']
    search_regions = search_config['search_regions']
    if len(window_sizes) != len(search_regions):
        print("ERROR: number of search_regions and number of window sizes don't match.")
        return None
    else:
        windows_list = []
        for i in range(len(window_sizes)):
            img, windows = find_cars(img, search_regions[i][0], search_regions[i][1], window_sizes[i], classifier)
            if draw == True:
                draw_boxes(img, windows)
            if len(windows) > 0 and type(windows) != None:
                windows_list.append(windows)

        if len(windows_list) > 0:
            windows = np.concatenate(windows_list)
        else:
            windows = np.array([])
        return img, windows





## Estimate Heat Map



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
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
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img



def heatmap_pipeline(image, bboxes, threshold=1):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    out_img = draw_labeled_bboxes(image, labels)
    return out_img


# ## For the video

# class video_pipe():

#     def __init__(self):
#         self.classifier = SVM.SVM_classifier()
#         print("Loaded classifier")
#         self.window_sizes = [1.0, 1.25, 1.5, 2.0]
#         self.search_regions = [[360, 600], [360, 600], [420, 720], [420, 720]]
#         self.search_config = {'window_sizes': self.window_sizes, 'search_regions': self.search_regions}
#         self.threshold = 2
#         self.previous_bboxes = []
#         self.counter = 0


#     def process_video(self, image):
#         # for the fist frame, use the basic heatmap
#         if self.counter < 1:
#             out_img, bboxes = (self.classifier).get_bboxes(image)
#             # out_img, bboxes = search_image(image, self.classifier, self.search_config)
#             out_img = heatmap_pipeline(image, bboxes, self.threshold)
#         else:
#             # out_img, bboxes = search_image(image, self.classifier, self.search_config)
#             out_img, bboxes = (self.classifier).get_bboxes(image)
#             # If current frame and prvious frame had bboxes, then apply a more robust heatmap
#             if len(bboxes) > 0 and len(self.previous_bboxes) > 0:
#                 present_bboxes = np.concatenate((bboxes, self.previous_bboxes), axis=0)
#                 out_img = heatmap_pipeline(image, present_bboxes, self.threshold+2 )
#             else:
#                 # If at least one of the two frames didnt have bboxes, then I apply the basic heatmap only using one of the methods
#                 if len(self.previous_bboxes) > 0:
#                     out_img = heatmap_pipeline(image, self.previous_bboxes , self.threshold)
#                 else:
#                     if len(bboxes) > 0:
#                         out_img = heatmap_pipeline(image, bboxes, self.threshold)
#                     else:
#                         out_img = np.copy(image)      

#         self.previous_bboxes = bboxes     
#         self.counter += 1
#         return out_img


class video_pipe():

    def __init__(self):
        self.classifier = SVM.SVM_classifier()
        print("Loaded classifier")
        self.window_sizes = [1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75]
        self.search_regions = [[340, 500],[440, 600], [425, 625], [400, 600], [420, 620], [550, 720], [520, 720]]
        self.search_config = {'window_sizes': self.window_sizes, 'search_regions': self.search_regions}
        self.threshold = 3
        self.previous_bboxes = []
        self.counter = 0


    def process_video(self, image):
        # for the fist frame, use the basic heatmap
        if self.counter < 1:
            out_img, bboxes = (self.classifier).get_bboxes(image)
            # out_img, bboxes = search_image(image, self.classifier, self.search_config)
            out_img = heatmap_pipeline(image, bboxes, self.threshold)
        else:
            # out_img, bboxes = search_image(image, self.classifier, self.search_config)
            out_img, bboxes = (self.classifier).get_bboxes(image)
            # If current frame and prvious frame had bboxes, then apply a more robust heatmap
            if len(bboxes) > 0 and len(self.previous_bboxes) > 0:
                present_bboxes = np.concatenate((bboxes, self.previous_bboxes), axis=0)
                out_img = heatmap_pipeline(image, present_bboxes, self.threshold+2 )
            else:
                # If at least one of the two frames didnt have bboxes, then I apply the basic heatmap only using one of the methods
                if len(self.previous_bboxes) > 0:
                    out_img = heatmap_pipeline(image, self.previous_bboxes , self.threshold)
                else:
                    if len(bboxes) > 0:
                        out_img = heatmap_pipeline(image, bboxes, self.threshold)
                    else:
                        out_img = np.copy(image)      

        self.previous_bboxes = bboxes     
        self.counter += 1
        return out_img