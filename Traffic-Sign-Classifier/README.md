# Traffic-Sign-Classifier
Udacity Second Project - Classify Traffic Signs using Neural Networks with Tensorflow

## Objective of the project 

The goal of this project was to classify traffic signs correctly by using neural networks and Tensorflow. More specifically, there were 6 objectives:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./DataDistribution.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./bicicle_pasing.jpg "Bicicle Passing"
[image5]: ./bumpy_road.jpg "Bumpy Road"
[image6]: ./mandatory_rotation.jpg "Mandatory Rotation"
[image7]: ./no_entry.jpg "No Entry"
[image8]: ./no_passing.jpg "No Passing"

Here is a link to my project (https://github.com/Rtamaki/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3]
* The number of unique classes/labels in the data set is 43

![Data Distribution][image1]

As we can see, the signs don't have equal amount of examples between the training, validation and test data sets. This difference in amount of images for each sign can, and probably does, cause different accuracy for each sign. Signs with fewer examples in the training data set will probably have lower accuracy than signs with many examples. 

Moreover, the proportion of sign images between datasets is also different, where one sign appears more often in the test dataset than it does in the validation( or training) data set. This different distribution of data signs contributes to differents accuracy between datasets (besides overfitting, of course).

## Model Architecture and Testing 

For preprocessing the images I restricted myself to simply normalize the data set, avoiding converting the images to grayscale. The normalization process is essential for numeric methods, as perfectly described in Udacity inntroduction class to neural networks, because of the precision limitations in computers. 

The normalization process has as an objective that the upper and lower bound limits for the values to be symmetric to zero, and preferably, with them being -1 and 1. In the case of RGB images, we have the lower bound limit at 0 and the upper at 255, so to normalize an image we have to substract all the values by 128 and then divide again by 128. Example:

[255, 0; 78, 169] -> [127, -128; -50, 41] -> [0.9921875, -1; 0.390625, 0.3203125]


With this process, we have 2 more decimal precision than before. This indeed proves to be an extremely important step, before I normalized the dataset the highest validation accuaracy I achieved was 0.875, but with just this preprocessing step, the precision jumped to 0.903.

The convertion to grayscale was advised but I decided not to do it, because I would loose information of the images, more exactly, I would have one third of the initial information after converting to grayscale, which could limit the precision of the neural network. Since the dataset isn't too big and the images are very small in size, the neural network can be quickly be trained, but was it otherwise, with high defition images or with a larger dataset, converting to grayscale could be a good compromise. 

As the last step, I added random rotation when feeding the validation and training data to make the neural network robust against rotated traffic signs. This increases the quality of the training data and adds also robusty against overfitting.

So the difference from the original data set are the random rotation when training and validating the neural network and the normalization.

### Neural Network Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5   | 1x1 stride, same padding, outputs 28x28x80 	|
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 14x14x80 				|
| Convolution 3x3   | 1x1 stride, same padding, outputs 12x12x120 	|
| RELU					    |												|
| Dropout           |     
| Convolution 3x3	  | 1x1 stride, same padding, outputs 10x10x150 	|
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 5x5x150 				|
| Fully connected		| output 400        									|
| RELU			    	  |
| Fully connected		| output 100        									|
| RELU			    	  |
| Fully connected		| output 84        									|
| RELU			    	  |
| Fully connected		| output 43       									|
| RELU			    	  |
 
My final neural network is very similar to LeNet, in the sense that at the beggining there are convolution layers and at the end fully connected layers. However, there are 3 differences:
1) Amount of layers: I have put 1 more convolution layer and 1 more fully connected layer. The additional convolutional layer permits the detection of more complex features that only 2 layers aren't able, as suggested in the neural network class, where we see the features detected in each layer.
2) Size of the layers: The neural network here has many more neuron than LeNet, which makes possible to detect more specific features. on the risk of over fitting.
3) Dropout layers: To avoid over fitting, 2 dropout layers were introduced, on in the convolutional layers and on at the fully connected layer. This way, the risk of overfitting is avoided and the neural network is more robust. Indeed, with the dropout out layers, the diffeence between the validation accuracy and test accuracy become much smaller.

The LeNet Neural Network was developed to recognize letters written in different forms, which involves recognizing small features such as curves and corners, then circles, ellipses, squares and triangles, and finnaly, combinations of the previous geometric forms. Because the traffic signs can also be decomposed in squares, triangles and ellipses, the LeNet architecture is a good start point.


### Training: 
For the training process, I made use of the same optimizer from the LeNet-Lab, "AdamOptimizer".
The size of the batch was limited to 2000, because I observed loss in validaion accuracy for when the batch was below 1000, so, just to be safe, I opted to define it at 2000 and still achieved satisfactory iteration time.

For the number of epochs and learning rates, I chose a method a bit different than used in Udacity's LeNet Lab. Instead of using a fixed learning rate and number of epochs, I made them variable, depending on how quickly the validation accuracy improves or on the achieved accuracy. I start the learning rate with the standard value of 0.001, which could prove to be too high and limit the accuracy, but I introduced an additional method, in which I decrease the learning rate depending on how quickly the validation accuracy improves. If it increases to slow, it is high likely caused because the learning rate is too large, therefore I lower it, and if it increases fast, than I can probably increase it even more. This way, I achieved fast training speed without having to compromise in precision.

The number of epochs defined is only the upper limit of epochs, because the main drive in any numerical optimization process is to achieve the required precision. Therefore, I iterated the neural network to depend on the accuracy achieved for the validation data set to be at least 0.96 (0.03 better than the required). With this modification, I guaranteed to achieve a minimum precision as fast as possible without taking the risk of taking to long, which usually mean that the implementation is wrong.

### Sufficiently adequate architecture

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.966
* test set accuracy of 0.900


To discover the adequate architecture I iterated several models with LeNet as base. The first architecture was exactly the same as LeNet, but there were some problems. First, the output had to be changed to 43 and the inputs [32, 32, 3] to accept color images. Additionaly, the architecture wasn't complex enough to be able to detect all the necessary features, so the amount of neurons in the deep layers also had to be modified. Indeed, the validation accuracy I achieved with LeNet was always below 0.90.

Therefore, it was clear that only using the pre estabilished LeNet architecture wasn't enough and I developed my own neural network with LeNet as base. The process took a ot of time, because the NN had to be trained for every modification I did, beginning with the most simple, by only changing the amount of neurons in each layer, then introducing more layers, and at last changing the overall architecture. The process was simple, as it only envolved changing the parameters and seeing the results, but was time consuming.

An important aspect to take account in every machine learning algorithm is the possibility of overfitting. To avoid that, I included 2 dropout layers to make sure both parts of the NN were robust against overfitting. Previous to the dropout, the validation and test accuracy were very distinct from each other, one sometime being 0.96 while the other was no more than 0.85.

However, even with these precautions, it is clear from the difference between the datasets accuracy that the neural networks is overfitting, since the difference between the validation accuracy and the test accuracy are over 0.06. So
 

## Test a Model on New Images

Here are five German traffic signs that I found on the web:

![][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I couldn't test properly for the web images because everytime the output was different, sometimes indicating for the same image different traffic signs.

The test on web images will have lower accuracy because:
1) On the training set the images are perfectly fit in the 32x32 frame, which almost never occurs. The traffic signs are at random places in the image. Therefore, to use simple convolutions isn't enough, since the image can have different scale and the features wont be detected. That is one of the reasons why the inception method is useful.
2) Multiple signs( or objects that resemble traffic signs): In real case scenarios, there can be several cases of false positives, where the image detects inexistent traffic signs, even when there is indeed an undetected one.
3) Obstruction: The dataset limit itself to perfectly uncovered traffic signs, but we would like to detect traffic signs even when they are partially covered but still recognisable. 


Here are the results of the prediction:

| Image			        |     Prediction	        					| Probability |
|:---------------------:|:---------------------------------------------:|:---------------------:| 
| Go Straight or Right      		|  Go Straight or Right								| 77.2% |
| Speed Limit 30km/h     			| Speed Limit 30km/h 										| 99.9% |
| Roundabout Mandatory					| Roundabout Mandatory											| 95.1% |
| No Entry      		| Priority Road  				 				| 100% |
| No Passing			| No Passing	      							| 100% |

The model accuracy for these web images is 80%, so it is another sign that the neural network is overfitting.It can be see that the model is almost always sure what the signs are, with the lowest still being 77.2%, which is still very high. But the model is certain what a sign is even though it is wrong.
There are 2 very clear reasons on why the model is underperforming:
1) The web images are distorced to fit the 32x32 frame
2) The size of the traffic signs in the image are different. As we are using only a fixed convolution, the precision of the neural network varies on how much of the 32x32 frame the sign occupies. This can be solved by increasing the training set with different sign sizes and/or using inception in the first layer of the neural network.
