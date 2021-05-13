# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



[//]: # (Image References)
[window_region1]: ./output_images/window_regions1.jpg
[window_region2]: ./output_images/windo_regions2.jpg
[window_region3]: ./output_images/window_regions3.jpg

[vehicle_detection1]: ./test1.jpg
[vehicle_detection2]: ./test2.jpg
[vehicle_detection3]: ./test3.jpg









The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The steps
----

### First Step: Build SVM classifier
The first step for the project is to build the classifier that will detect in an image/video frame where the vehicles are located. To do so we build a linear SVM classifier with the images from the folders 'vehicles' and 'non-vehicles'. The classifier won't use the images directly, but will use instead a color histogram and the Histogram of Gradients(HoG) which are the so called preprocessing. After the preprocessing, the classifier has less features at disposal which speeds up the calcultions, but may also unable the classifier to work, since we have less information than before. The proprocessing is pretty much a transformation in the dimensions of the features to have a diferent hyperspace than at the begggining, which is equivalent to use a non-linear kernel for the SVM. 
  In the case of artificial inteligence for image classification we can use ourselves to decide which transformations in the preprocessing step, since humans are good image classifiers(and we take ourselves as the best image classifier yet). This intuition of which transformation may be usefull may not work in other more abstract fields and may also lead to a more complicated transformation than necessary(for the SVM) and/or more inneficient. So we have always to be careul, since intuition may be a source for scientific discveries but isn't science in itself.
  The steps to properly train the classfier are

  1) Calulate the color histogram
  2) Calculate the Histogram of Gradients
  3) Combine the two histograms to create a feature for each image
  4) Select all the images in the folder 'vehicles' and set them as having y = 1
  5) Selct all the images in the folder 'non-vehicles' and given them label = 0
  6) Create the training and test features & labels sets(or use the cross-validation pakage from sklearn)
  7) Use the package StandardScaler from sklearn.preprocessing to normalize the features
  8) Train the classifier
  9) Save the classifier for later use
  
  The last step is specially critical: During the development of the program it becomes very easy to forget how one trained the SVM classifier. Since when we will pass the images/video to search or cars we will need to use the *exact* same configurations used to train the classifier, if we save the configurations used, we guarantee no problems will occur because of theuse of different configrations. Even worse than error appearing when running the programm, is when the programm runs but the accuracy and results are worse than expected. One example that appeared frequently was the use of different color spaces, which made the classifier completely useless. Alas, it is a good practice to make the programm robust against such errors.
  
  After testing several combinations of color spaces, number of spatial bins and number of color bins, I selected the current configuration because not only it was the best from among all of the tested, but also because the prediction time was satisfactory.
* *Color Space*: YCrCb
* *Number of Spatial Bins*: 32
* *Number of Color Bins*: 32
* *Pixels per cell*: 16
* *Cells per block*: 2
* *Number of possible orientations*: 9
* *Number of channels for the HoG*: All 3
  
### Second Step: Create a process to apply the image classifier SVM for an image
Now that we have the SVM classifier we want to apply it for the image in the folders in the folder 'test_images'. Here we will:
  1) Define te region of the image to search for cars, since they can only appear in the lower half of the image.
  2) Define windows sizes and how many different sizes that will search for cars. Here we can be even more efficient and define search regions dependent to the window sizes. Here I chose to use 4 but since it appeared to not be too many(wich would slow down the process) but also not too few, which could lower the accuracy
  2.1) More exactly, the search region will be at most the lower half of the image, but for each different window size we will use different regions, since small objects (i.e. small cars) appear closer to the horizon and bigger objects appear closer to end of the image. This process not only improve the processing speed, but also reduces the number of false positives. As an example, we can see from the images below that indeed some window sizes are only suitable to be applied near the horizon while others should be applied far from it.
    ![Window Sizes and Search Region  1][window_region1]
    ![Window Sizes and Search Region 2][window_region2]
    ![Window Sizes and Search Region 3][window_region3]
    
    For this project I decided in using window sizes of 64x64, 96x96 and 128x128 pixels.
  3) Make the SVM classify all the windows. Here we can optimize even further by calculating the HoG only once instead of multiple times.
    3.1) In th normal approach, depicted in *lesson_function.detect_vehicle* we select a windowed image, resize it to the same shape as the images used to train the SVM and then calculate the features. However, this is highly inneficient, since the windows overlap and we end up calculating features HoG for the same pixels over and over. Instead of calculating the HoG features for each window, we calculate for HoG for all cells/blocks fo the whole image, and then select only the blocks that would correspond to the window.
    Taking as an example a window os size 64x64 pixels, an overlaping of 50%, and a search region of 180x1080 pixels, the twould need more than 3 time more calculations for the HoG features using the traditional method in comparision to the more efficient one.
    The usual approach to calculate the HoG features for each image are implemented in the file *lesson_functions* in the functions *detect_vehicle*, *search_windows* and *single_img_features*. The function *detect_vehicle* searches for each window size the image/frame by calling the function *search_windows*.
    The better method is implemented in the function *find_cars*, which we can see very clearly that we calculate the HoG features for the subregion of interest only once. 
  4) It becomes clear after testing that the search regions for different window sizes can't be mutually exclusive: some overlap is necessary to guarante that all the vehicles are detected. In the 3 images below, we see that the algorithm detected the black car for all different window sizes, but only for one windw size was the white car detected.
  
  ![Window Sizes and Search Result 1][vehicle_detection1]
  ![Window Sizes and Search Result 2][vehicle_detection2]
  ![Window Sizes and Search Result 3][vehicle_detection3]
  
  5) No that we have all the bounding boxes we need to combine overlping boundig boxes as one. We do that by applying the concept of a heat map, which basically is an indication of how certain the algorithm is that we identified a vehicle in a given region: the more bounding boxes, the greater the heat, and , therefore, the greater the likehood of a vehicle. Here are the steps to implement the heatmap.
    5.1) Create a black image with the same size of the original image
    5.2) We add 1 to all the pixesls inside a bounding boxes for each bounding box the pixels is in. This is the *heat*.(function *add_heat*)
    5.3) Threshold the image to select only the pixels with value above te given threshold (function *apply_threshold*)
    5.4) Now we have a black and white image. Using sci-kit-learn library's function *label* we get bounding boxes that contains *all* the white pixel regions. Now instead of multiple bounding boxes for a single vehicle, we have a bounding box which contains(depend on the value of the threshold selected) all the bounding boxes for a single vehicle.
    
  
  
We need to be careful, since SVM that we trained before only works for a given size of image, since the number of HoG features depends on the size of the image we feed. Therefore, before feeding the windowed image, we need to either resize to the format user for training or create an SVM classifier for each image size. Since the last option would require us to retrain several classifiers each time we choose to use different window sizes, I chose for resizing the window image instead.

In total, when applying the vehicle detection algorithm I took particular attention to:
 1) The number of features: the processing time is directly correlated to th number of features. After much experimentation, I decided for the described confgurations from before since the processing time was bearable andthe actual accuracy satisfactoy.
 2) The search region for each window size: The search region defines the ammount of slinding windows for each window size, which has direct influence to the number of calculations done. I decided to keep them as narrow as possible:
 
 
| Window Size         		|     (ymin, ymax)	        					| 
|:---------------------:|:---------------------------------------------:| 
| 64 x 64         		| (420, 520)							| 
| 96 x 96   | (400,560) 	|
| 128 x 128			    |				(400, 600)								|

3) Reuse the calcultions of the HoG features to avoid unnecessary calculations: This is one of the most important improvements since there is no cost and we improve the speed by almost a 3x factor.


### Third step: apply the algorithm for a video:
  This is the last part and is almost how the algorithm would work in the field. However, it is very likely that in *some* frames of the video the algorithm gives *false positives*. What we can do is to use previous informations from past frames to filter out such cases, since we know that vehicles won't probably move so fast that the bounding bouxes of consecutive frames won't overlap.
  For this reason, in addition the the bouding boxes(before the heatmap) for the current frame we can use the previous bounding boxes and apply the past and current bounding boxes in the heatmap algorithm. Doing this we can select a higher value for the heatmap's threshold and filter out the false positives.(Done inside the class *video_pip*)
  Furthermore, it is important to load the classifier only once per video, instead then once per frame. For this reason, I createda *class* called *video_pipe* which contains the classifier as an static variable for all the frames.
  
 
 #### Discussion
 ----
 From this project, it became clear of the bottleneck when trying to identify vehicles in the road and the importance of optimizing the calculations to achieve algorithms that could work in real time. To do so, we must not spend processing time in features that can be reused or in those that don't influence the accuracy of the algorithm.

Some points that were not taken in consideration in this project:
1) The algorithm only trained to identify if a window contained a vehicle or not. More precisely, if it contained cars. Therefore, the algorithm most certainly wont have the same precision as the one achieved when other vehicles, such as buses, motorbicicles and trucks are in the scene. So our algorithm only is applicable for a subgroup of the actual situations a normal car would confront.
2) From the discussion from the previous topic, the algorithm doen't identify vehicles other than cars. If we decided to train a SVM to include these vehicles, it may still not be enough, because we may want that our autonomous vehicle to be behave differntly for bicicles than for buses. If this happens, the usual SVM won't be enough and we would need to use a multicass SVM like one-against-all or one-against-one or DAGSVM or even another machine learning algorithm.
3) Again, we hve the weather conditions in play. In situations when visibility is poor, the programm may not idetify vehicles a bit further ahead and let the autonomous vehicle behave improperly. We could train multiple distinct SVMs in which each would be used for different visibility conditions. It is, of course, possible to use only one and train it for all visibilities cases, but it is very likely that this case would be worse.
4) At last, a discussion of the process of using several preprocessing steps before feeding the SVM with features. All the preprocessing steps we have applied, HoG, color space trasnformations, histogram of colors, are all linear and no linear combination of the features we had before, which was the RBG imge. So, the preprocessing we did was pretty much a mapping function transformation from the original space to a higher degree space to obtain a non-linear kernel for the SVM. Therefore, if we useed a RBF(~radial function) it is possible we would achieve better accuracy but woulld loose a lot in speed.
5) Since the preprocessing was resposible for most of the time spent in each window( almost 9x more than the prediction), it may be useful to think in other algorthms to apply
6) Since the prediction for each window are independent from other window predictions, we run the predictions for multipl widnows in parallel and even use the GPU the speed up.

 
