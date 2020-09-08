#**Traffic Sign Recognition** 

##Writeup Report


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_label_counts_in_training_set.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/bicycles_crossing_29.jpg "Traffic Sign 1"
[image5]: ./new_images/children_crossing_28.jpg "Traffic Sign 2"
[image6]: ./new_images/dangerous_curve_to_right_20.jpg "Traffic Sign 3"
[image7]: ./new_images/bumpy_road_22.jpg "Traffic Sign 4"
[image8]: ./new_images/traffic_signals_26.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

###Data Set Loading
I downloaded the German Traffic Signs dataset from the link mentioned in Project Instructions. This contained the following pickle files -

 * train.p for training set
 * valid.p for validation set
 * test.p for test set
 
 I loaded these files in the code using the pickle module. The code for this step is in first code cell of IPython notebook 
 
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy and pandas libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is  34799
* The size of test set is  12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43
* Training Set does not contain equal number of images/examples for different classes. SignNames that have  highest number of examples and lowest number of examples are printed.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the counts for each class label from the training set...


![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in 4th, 5th, 6th, 7th cells of the IPython notebook.

I defined a set of methods that are useful for preprocessing which are available in 4th cell. They include rescaling the data, standardization of features (zero centering and unit variance), conversion to grayscale and conversion to YCrCb.
I defined a preprocessing pipeline in 5th cell that can be used to apply differrent kinds of preprocessing methods on input data such as X_train. This takes a list of functions so we can specify all the functions we want to apply as part of preprocessing
In code cell 6th, I appplied preprocessing on all three datasets (X_train, X_valid and X_test).
I have defined utility function in code cell 7th for visualizing few images (i.e preprocessed images) from all three datasets.

As a first step, I rescaled the images to 0.0 to 1.0 float values and then applied gray scale conversion and zero centering. Though color definitly has great information, I believed that it is the objects shape and content within the image that is more useful on this dataset for classification and hence grayscale conversion. As part of normalization I performed mean subtraction to zero-center the data. I did not divide the features by standard deviation (my accuracy went down when I used it so I disabled that part)


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Code cell 6 produced final X_train, X_valid and X_test datasets (after preprocessing). The sizes of these are:
X_train contained 34799 examples
X_valid contained 4410 examples
X_test contained 12630 examples.
I did not get a chance to explore data augmentation techniques such as randomly cropping, shifting, rotating, adjusting brightness/contrast etc.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Code cell 8, 9, 10, 11, 12 are related to my model architecture, configuration and training with differnet configurations and finally plotting the validation accuracy results.
I defined the model architecture in Cell 8 - I started with LeNet5 and modified it to include Dropout.
I defined various operations, placeholders in TensorFlow in code cell 9.
I defined different configurations in code cell 10. Each configuration describes a specific value for different hyperparameters such as batch size, learning rate etc
Code cell 11 runs my training loop where this was repeated for all chosen configurations.
Validation accuracy results are plotted in Cell 12


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU  | Used Exponential Linear Units as the Activation function					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Dropout | For a better generalization capabilities of the model 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16.      |									|
| ELU | Used Exponential Linear Units as the Activation function |
| Dropout | For a better generalization capabilities of the model|
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Flattened | Used tf.contrib.layers.flatten for this purpose
| Fully connected		| Input is 400, output = 120
| Fully connected | Number of Input neurons = 120, Number of output neurons = 84 |        									|
| Softmax				| Number of input neurons = 84, Number of output neurons (or classes) = 43       									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is in code cell 11 - I used this cell for identifying the best set of hyperparameters. Then I used code cell 13 for training the final model with the best configuration identified and with number of epochs set to 100. The model is also saved to a file in this cell. 

To train the model, I used AdamOptimizer with a batch size of 50, learning rate of 0.001, dropout keep probability of 0.7.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the Test accuracy of the model is located in the 14th code cell of the Ipython notebook. I chose different values for hyperparameters and trained multiple times and picked the  set of values that provided better accuracy. I used validation set in determining that.

My final model results were:
* training set accuracy of 98.6 (from code cell 15)
* validation set accuracy of 95.4 (from code cell 13)
* test set accuracy of 92.9 (from code cell 14)


If a well known architecture was chosen:
* What architecture was chosen?
I started with LeNet5 and modified it to include Dropout as it provides good regularization effect. I also used ELU instead of RELU
* Why did you believe it would be relevant to the traffic sign application?
LeNet5 worked well for MNIST dataset which has around 10 classes. This traffic dataset contained 43 classes and I believed the model is powerful enough (as it contained convolutions, pooling (R)ELU, and dropout) to be able to correctly discriminate among differnet classes.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation set is used in adjusting the models hyperparameter - so it directly influenced how I was tweaking different hyperparameters. I did not use test set to measure the accuracy until I hit an accuracy value > 90% on validation set. Though the accuracy on my test set is a couple of points down compared to validation set, it was still above 90% so I believe it should generalize well
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The training set does not contain equal number of images for each sign. So, as part of my new images, I chose the signs where the number of images seen in the training set is low.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is in code cells 16, 17, 18 and 19 of the Ipython notebook.
Code cell 16 loads and displays the image and resizes them to 32x32
Code cell 17 collects the expected target values for these new images.
Code cell 18 applies preprocessing pipeline on the new images
Code cell 19 runs the model to predict the traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycles crossing      		| Bicycles crossing   									| 
| Children crossing    			| Children crossing 										|
| Danger curve to right					| Dangerous curve to right											|
| Bumpy road      		| Bumpy Road					 				|
| Traffic signals 			| Traffic Signals      							|


The model was able to correctly guess 5 out of the 5 traffic signs, which gives an accuracy of 100%. This is higher than what I expected based on test set accuracy. A dataset size of 5 new samples may be too small in deciding how it compares against test accuracy but it provided great opporutnity to try everything on new images (such as loading data, resizing, preprocessing, running model for predictions and evaluating accuracy) 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for getting top 5 predictions on my final model is located in the 20th code cell of the Ipython notebook.
The top-5 prediction probabilites along with their sign names are shown below. Each row first shows the sign names for a given target label and the row immediately below that shows the probabilities for the same target label

| Target Class | Prediction 1 | Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5|
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|Bicycles crossing | Bicycles crossing | Speed limit (60km/h) | Slippery Road | Ahead only | Bumpy Road|
|Bicycles crossing | 0.99886537 |  0.00068288|  0.00040369|  0.00003467|  0.00001113|
|Children crossing | Children crossing | Dangerous curve to the left | Right-of-way at the next intersection | Beware of ice/snow | Dangerous curve to the right|
|Children crossing | 1.     |     0.00000001 |  0.00000001 |  0. |          0.      |
|Dangerous curve to the right|Dangerous curve to the right|  Slippery road|End of no passing|Dangerous curve to the left|End of no passing by vehicles over 3.5 metric|
|Dangerous curve to the right|0.99753076 | 0.00220433 |  0.00022909 |  0.00003039 |  0.00000536|
|Bumpy road|Bumpy road| Bicycles crossing | Road work | Traffic signals | Children crossing |
|Bumpy road|0.99999881 | 0.00000083 |  0.00000038 |  0. |          0.  |
|Traffic signals|Traffic signals|General caution | Pedestrians|Right-of-way at the next intersection | Beware of ice/snow|
|Traffic signals|0.99974483 | 0.00025505 |  0.00000013 |  0.00000003 |  0.      |
